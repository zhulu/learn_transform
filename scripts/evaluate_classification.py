from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classification_data import TextClassificationDataset, make_classification_collate_fn
from src.classification_inference import load_classifier_checkpoint
from src.classification_training import run_epoch


def main() -> None:
    """在 AG News 测试集上计算 loss 和 accuracy。"""
    parser = argparse.ArgumentParser(description="Evaluate an AG News classifier checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="dataset/ag_news")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    model, vocab, labels, config, device = load_classifier_checkpoint(args.checkpoint, args.device)
    dataset = TextClassificationDataset(
        Path(args.data_root) / f"{args.split}.csv",
        vocab,
        max_len=config.max_len,
        lowercase=config.lowercase,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_classification_collate_fn(vocab.pad_id),
        pin_memory=device.type == "cuda",
    )
    loss, acc = run_epoch(
        model,
        loader,
        nn.CrossEntropyLoss(),
        device,
        optimizer=None,
        scaler=None,
        use_amp=False,
        grad_clip=0.0,
        desc=f"eval {args.split}",
    )
    print(f"split={args.split} labels={labels} examples={len(dataset)} loss={loss:.4f} acc={acc:.4f}")


if __name__ == "__main__":
    main()
