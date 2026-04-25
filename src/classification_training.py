from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .classification_data import (
    AG_NEWS_LABELS,
    TextClassificationDataset,
    build_vocab_from_classification_file,
    make_classification_collate_fn,
)
from .classifier import TransformerTextClassifier
from .training import resolve_device
from .vocab import Vocabulary


@dataclass
class ClassificationConfig:
    """文本分类训练配置。"""

    train_csv: str
    test_csv: str
    save_dir: str = "checkpoints/ag_news_classifier"
    vocab_size: int = 30000
    min_freq: int = 2
    max_len: int = 128
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 3
    d_ff: int = 1024
    dropout: float = 0.1
    batch_size: int = 128
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-4
    valid_ratio: float = 0.1
    num_workers: int = 0
    seed: int = 42
    lowercase: bool = True
    amp: bool = True
    grad_clip: float = 1.0
    device: str = "auto"


def set_seed(seed: int) -> None:
    """固定随机种子，方便复现实验。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_classifier(config: ClassificationConfig, vocab: Vocabulary) -> TransformerTextClassifier:
    """根据配置创建文本分类模型。"""
    return TransformerTextClassifier(
        vocab_size=len(vocab),
        pad_id=vocab.pad_id,
        num_classes=len(AG_NEWS_LABELS),
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
        max_len=config.max_len + 8,
    )


def count_parameters(model: nn.Module) -> int:
    """统计模型可训练参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_classification(config: ClassificationConfig) -> None:
    """完整的文本分类训练流程。"""
    set_seed(config.seed)
    device = resolve_device(config.device)
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = save_dir / "vocab.json"
    if vocab_path.exists():
        vocab = Vocabulary.load(vocab_path)
    else:
        vocab = build_vocab_from_classification_file(
            config.train_csv,
            vocab_size=config.vocab_size,
            min_freq=config.min_freq,
            lowercase=config.lowercase,
        )
        vocab.save(vocab_path)

    full_train = TextClassificationDataset(
        config.train_csv,
        vocab,
        max_len=config.max_len,
        lowercase=config.lowercase,
    )
    test_set = TextClassificationDataset(
        config.test_csv,
        vocab,
        max_len=config.max_len,
        lowercase=config.lowercase,
    )

    valid_size = int(len(full_train) * config.valid_ratio)
    train_size = len(full_train) - valid_size
    generator = torch.Generator().manual_seed(config.seed)
    train_set, valid_set = random_split(full_train, [train_size, valid_size], generator=generator)

    collate_fn = make_classification_collate_fn(vocab.pad_id)
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    model = create_classifier(config, vocab).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    meta = {
        "config": asdict(config),
        "labels": AG_NEWS_LABELS,
        "vocab_size": len(vocab),
        "train_examples": len(train_set),
        "valid_examples": len(valid_set),
        "test_examples": len(test_set),
        "parameters": count_parameters(model),
    }
    with (save_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))

    best_valid = math.inf
    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            grad_clip=config.grad_clip,
            desc=f"train {epoch}/{config.epochs}",
        )
        valid_loss, valid_acc = run_epoch(
            model,
            valid_loader,
            criterion,
            device,
            optimizer=None,
            scaler=None,
            use_amp=False,
            grad_clip=0.0,
            desc=f"valid {epoch}/{config.epochs}",
        )
        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"valid_loss={valid_loss:.4f} valid_acc={valid_acc:.4f}"
        )

        checkpoint = {
            "model": model.state_dict(),
            "config": asdict(config),
            "labels": AG_NEWS_LABELS,
            "epoch": epoch,
            "valid_loss": valid_loss,
            "valid_acc": valid_acc,
        }
        torch.save(checkpoint, save_dir / "last.pt")
        if valid_loss < best_valid:
            best_valid = valid_loss
            torch.save(checkpoint, save_dir / "best.pt")

    # 最终测试时加载验证集上最好的 checkpoint，而不是直接使用最后一个 epoch。
    best_path = save_dir / "best.pt"
    if best_path.exists():
        best_payload = torch.load(best_path, map_location=device)
        model.load_state_dict(best_payload["model"])

    test_loss, test_acc = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        optimizer=None,
        scaler=None,
        use_amp=False,
        grad_clip=0.0,
        desc="test",
    )
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")


def run_epoch(
    model: TransformerTextClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
    grad_clip: float,
    desc: str,
) -> tuple[float, float]:
    """运行一个分类任务 epoch，返回平均 loss 和准确率。"""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    iterator = tqdm(loader, desc=desc)
    for input_ids, labels in iterator:
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(input_ids)
                loss = criterion(logits, labels)

            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

        preds = logits.argmax(dim=-1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_examples += batch_size
        iterator.set_postfix(
            loss=total_loss / max(1, total_examples),
            acc=total_correct / max(1, total_examples),
        )

    return total_loss / max(1, total_examples), total_correct / max(1, total_examples)
