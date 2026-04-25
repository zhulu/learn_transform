from __future__ import annotations

import argparse
from pathlib import Path
import sys

# 让脚本可以直接导入 src。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classification_inference import classify_text, load_classifier_checkpoint


def main() -> None:
    """单句新闻分类入口。"""
    parser = argparse.ArgumentParser(description="Classify one news text with a trained checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/ag_news_classifier/best.pt")
    parser.add_argument("--text", required=True)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    model, vocab, labels, config, device = load_classifier_checkpoint(args.checkpoint, args.device)
    label, prob, scores = classify_text(
        model,
        vocab,
        labels,
        args.text,
        device,
        max_len=args.max_len,
        lowercase=config.lowercase,
    )
    print(f"label={label} prob={prob:.4f}")
    for name, score in scores:
        print(f"{name}: {score:.4f}")


if __name__ == "__main__":
    main()
