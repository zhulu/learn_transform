from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import ParallelTextDataset, make_collate_fn
from src.inference import load_checkpoint
from src.training import run_epoch


def main() -> None:
    """在翻译测试集上计算 loss 和 perplexity。"""
    parser = argparse.ArgumentParser(description="Evaluate a translation checkpoint on a parallel split.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--src-lang", required=True)
    parser.add_argument("--tgt-lang", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    model, src_vocab, tgt_vocab, config, device = load_checkpoint(args.checkpoint, args.device)
    data_root = Path(args.data_root)
    dataset = ParallelTextDataset(
        data_root / f"{args.split}.{args.src_lang}",
        data_root / f"{args.split}.{args.tgt_lang}",
        src_vocab,
        tgt_vocab,
        max_len=config.max_len,
        lowercase=config.lowercase,
        src_char_level=config.src_char_level,
        tgt_char_level=config.tgt_char_level,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_collate_fn(src_vocab.pad_id, tgt_vocab.pad_id),
        pin_memory=device.type == "cuda",
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_id, label_smoothing=0.0)
    loss = run_epoch(
        model,
        loader,
        criterion,
        device,
        optimizer=None,
        scaler=None,
        use_amp=False,
        grad_clip=0.0,
        desc=f"eval {args.split}",
    )
    print(f"split={args.split} examples={len(dataset)} loss={loss:.4f} ppl={math.exp(loss):.4f}")


if __name__ == "__main__":
    main()
