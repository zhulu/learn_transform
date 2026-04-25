from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chinese_sentiment_inference import (
    load_chinese_sentiment_checkpoint,
    predict_chinese_sentiment,
)


def main() -> None:
    """中文情感分类推理入口。"""
    parser = argparse.ArgumentParser(description="Predict sentiment for one Chinese text.")
    parser.add_argument("--checkpoint", default="checkpoints/chnsenticorp/best.pt")
    parser.add_argument("--text", required=True)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    model, vocab, labels, config, device = load_chinese_sentiment_checkpoint(args.checkpoint, args.device)
    label, prob, scores = predict_chinese_sentiment(
        model,
        vocab,
        labels,
        args.text,
        device,
        max_len=args.max_len,
        lowercase=config.lowercase,
        char_level=config.char_level,
    )
    print(f"label={label} prob={prob:.4f}")
    for name, score in scores:
        print(f"{name}: {score:.4f}")


if __name__ == "__main__":
    main()
