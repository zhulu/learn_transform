from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chinese_sentiment_training import ChineseSentimentConfig, train_chinese_sentiment


def main() -> None:
    """中文情感分类训练入口。"""
    parser = argparse.ArgumentParser(description="Train a Transformer encoder on ChnSentiCorp.")
    parser.add_argument("--data-root", default="dataset/chnsenticorp")
    parser.add_argument("--save-dir", default="checkpoints/chnsenticorp")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--word-level", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    config = ChineseSentimentConfig(
        train_tsv=str(data_root / "train.tsv"),
        valid_tsv=str(data_root / "valid.tsv"),
        test_tsv=str(data_root / "test.tsv"),
        save_dir=args.save_dir,
        vocab_size=args.vocab_size,
        min_freq=args.min_freq,
        max_len=args.max_len,
        d_model=args.d_model,
        num_heads=args.heads,
        num_layers=args.layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_workers=args.num_workers,
        device=args.device,
        amp=not args.no_amp,
        char_level=not args.word_level,
    )
    train_chinese_sentiment(config)


if __name__ == "__main__":
    main()
