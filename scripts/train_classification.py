from __future__ import annotations

import argparse
from pathlib import Path
import sys

# 让脚本从项目根目录运行时能找到 src 包。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classification_training import ClassificationConfig, train_classification


def main() -> None:
    """文本分类训练入口。"""
    parser = argparse.ArgumentParser(description="Train a Transformer encoder text classifier.")
    parser.add_argument("--data-root", default="dataset/ag_news")
    parser.add_argument("--save-dir", default="checkpoints/ag_news_classifier")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    config = ClassificationConfig(
        train_csv=str(data_root / "train.csv"),
        test_csv=str(data_root / "test.csv"),
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
        valid_ratio=args.valid_ratio,
        num_workers=args.num_workers,
        device=args.device,
        amp=not args.no_amp,
    )
    train_classification(config)


if __name__ == "__main__":
    main()
