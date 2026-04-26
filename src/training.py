from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import ParallelTextDataset, build_vocabs_from_files, make_collate_fn
from .model import Transformer
from .vocab import Vocabulary


@dataclass
class TrainConfig:
    """训练配置。

    这个 dataclass 把命令行参数、模型超参数、数据路径集中在一起。
    好处是 checkpoint 里可以直接保存配置，推理时能用同一份配置恢复模型结构。
    """

    # 训练集和验证集的平行语料路径。
    train_src: str
    train_tgt: str
    valid_src: str
    valid_tgt: str
    # checkpoint、词表和 config.json 的保存目录。
    save_dir: str = "checkpoints/translation"
    # 词表参数：词表越大，embedding 和输出层参数越多。
    src_vocab_size: int = 16000
    tgt_vocab_size: int = 16000
    min_freq: int = 2
    # 过滤超过 max_len 的句子，避免显存被超长样本拖垮。
    max_len: int = 128
    # Transformer 结构参数。
    d_model: int = 256
    num_heads: int = 4
    encoder_layers: int = 3
    decoder_layers: int = 3
    d_ff: int = 1024
    dropout: float = 0.1
    # 训练参数。
    batch_size: int = 128
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    num_workers: int = 0
    seed: int = 42
    # 分词参数。中文端可以先设 char_level=True。
    lowercase: bool = True
    src_char_level: bool = False
    tgt_char_level: bool = False
    # amp=True 时，在 CUDA 上使用混合精度，3090 会更省显存也更快。
    amp: bool = True
    grad_clip: float = 1.0
    device: str = "auto"


def set_seed(seed: int) -> None:
    """固定随机种子，让实验结果尽量可复现。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    """解析训练设备。

    name="auto" 时，如果能检测到 CUDA 就用 GPU，否则回退到 CPU。
    """
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def create_model(config: TrainConfig, src_vocab: Vocabulary, tgt_vocab: Vocabulary) -> Transformer:
    """根据配置和词表创建 Transformer。

    词表大小决定 embedding 和最终分类层大小；pad_id 决定 mask 逻辑。
    """
    return Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        src_pad_id=src_vocab.pad_id,
        tgt_pad_id=tgt_vocab.pad_id,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_encoder_layers=config.encoder_layers,
        num_decoder_layers=config.decoder_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
        max_len=config.max_len + 8,
    )


def count_parameters(model: nn.Module) -> int:
    """统计可训练参数量，用来评估模型大小。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_translation(config: TrainConfig) -> None:
    """完整训练入口。

    流程：
    1. 固定随机种子并确定设备。
    2. 构建或加载词表。
    3. 构建 Dataset/DataLoader。
    4. 创建模型、优化器、loss。
    5. 循环训练多个 epoch，并保存 last/best checkpoint。
    """
    set_seed(config.seed)
    device = resolve_device(config.device)
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    src_vocab_path = save_dir / "src_vocab.json"
    tgt_vocab_path = save_dir / "tgt_vocab.json"
    if src_vocab_path.exists() and tgt_vocab_path.exists():
        # 如果目录里已有词表，继续使用旧词表，保证恢复训练/推理时 id 映射一致。
        src_vocab = Vocabulary.load(src_vocab_path)
        tgt_vocab = Vocabulary.load(tgt_vocab_path)
    else:
        # 只用训练集建词表，避免验证集信息泄漏进训练流程。
        src_vocab, tgt_vocab = build_vocabs_from_files(
            config.train_src,
            config.train_tgt,
            src_vocab_size=config.src_vocab_size,
            tgt_vocab_size=config.tgt_vocab_size,
            min_freq=config.min_freq,
            lowercase=config.lowercase,
            src_char_level=config.src_char_level,
            tgt_char_level=config.tgt_char_level,
        )
        src_vocab.save(src_vocab_path)
        tgt_vocab.save(tgt_vocab_path)

    # Dataset 内部会完成分词、长度过滤、token id 编码。
    train_set = ParallelTextDataset(
        config.train_src,
        config.train_tgt,
        src_vocab,
        tgt_vocab,
        max_len=config.max_len,
        lowercase=config.lowercase,
        src_char_level=config.src_char_level,
        tgt_char_level=config.tgt_char_level,
    )
    valid_set = ParallelTextDataset(
        config.valid_src,
        config.valid_tgt,
        src_vocab,
        tgt_vocab,
        max_len=config.max_len,
        lowercase=config.lowercase,
        src_char_level=config.src_char_level,
        tgt_char_level=config.tgt_char_level,
    )

    # collate_fn 负责把不同长度样本 padding 成同一个 batch 张量。
    collate_fn = make_collate_fn(src_vocab.pad_id, tgt_vocab.pad_id)
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

    # 模型、优化器和损失函数。
    model = create_model(config, src_vocab, tgt_vocab).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    # ignore_index 忽略 <pad> 位置的 loss；label_smoothing 能缓解模型过度自信。
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_id, label_smoothing=config.label_smoothing)
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 把本次实验元信息写入 config.json，便于之后复现实验。
    meta = {
        "config": asdict(config),
        "src_vocab_size": len(src_vocab),
        "tgt_vocab_size": len(tgt_vocab),
        "train_examples": len(train_set),
        "valid_examples": len(valid_set),
        "parameters": count_parameters(model),
    }
    with (save_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))

    best_valid = math.inf
    for epoch in range(1, config.epochs + 1):
        # 训练阶段：会执行 backward 和 optimizer.step。
        train_loss = run_epoch(
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
        # 验证阶段：只前向计算 loss，不更新参数。
        valid_loss = run_epoch(
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
        print(f"epoch={epoch} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f}")

        checkpoint = {
            "model": model.state_dict(),
            "config": asdict(config),
            "src_vocab_size": len(src_vocab),
            "tgt_vocab_size": len(tgt_vocab),
            "epoch": epoch,
            "valid_loss": valid_loss,
        }
        # last.pt 始终保存最近一次 epoch 的模型。
        torch.save(checkpoint, save_dir / "last.pt")
        if valid_loss < best_valid:
            # best.pt 保存验证集 loss 最低的模型，推理通常加载它。
            best_valid = valid_loss
            torch.save(checkpoint, save_dir / "best.pt")


def run_epoch(
    model: Transformer,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    grad_clip: float = 0.0,
    desc: str = "",
) -> float:
    """运行一个 epoch。

    当 optimizer 不为 None 时是训练模式，否则是验证模式。

    batch 中的 tgt 形如：
        [<bos>, w1, w2, ..., <eos>]

    训练时要做 teacher forcing：
    - tgt_in:  [<bos>, w1, w2, ...]
    - tgt_out: [w1,    w2, w3, ..., <eos>]

    模型看到 tgt_in 的当前位置及之前 token，预测 tgt_out 的对应位置。
    """
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_tokens = 0
    iterator = tqdm(loader, desc=desc)
    for src, tgt in iterator:
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        # decoder 输入去掉最后一个 token；标签去掉第一个 <bos>。
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        if training:
            # set_to_none=True 通常比把梯度清零更省一点内存和时间。
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            # CUDA 上启用 autocast 后，部分矩阵计算会用 fp16/bf16，加速并省显存。
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(src, tgt_in)
                # CrossEntropyLoss 需要二维 logits: [N, vocab_size] 和一维标签 [N]。
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            if training:
                assert scaler is not None
                # AMP 下先缩放 loss 再反向传播，降低 fp16 梯度下溢风险。
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    # 裁剪梯度可以避免训练早期梯度爆炸。
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()

        # 用非 pad token 数量做加权平均，避免短 batch 和长 batch 权重不一致。
        ntokens = (tgt_out != model.tgt_pad_id).sum().item()
        total_loss += loss.item() * ntokens
        total_tokens += ntokens
        # 加上当前学习率显示
        lr_val = optimizer.param_groups[0]["lr"] if optimizer else 0.0
        iterator.set_postfix(loss=total_loss / max(1, total_tokens), lr=f"{lr_val:.2e}")

    return total_loss / max(1, total_tokens)
