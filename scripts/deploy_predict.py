from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.chinese_sentiment_inference import (
    load_chinese_sentiment_checkpoint,
    predict_chinese_sentiment,
)
from src.classification_inference import classify_text, load_classifier_checkpoint
from src.inference import greedy_translate, load_checkpoint


TASK_MODEL_DIRS = {
    "en_de_translation": "multi30k_en_de",
    "en_zh_translation": "tatoeba_en_zh",
    "ag_news": "ag_news_classifier",
    "zh_sentiment": "chnsenticorp",
}


def checkpoint_for(package_root: Path, task: str) -> Path:
    """返回部署包中某个任务的 best.pt 路径。"""
    return package_root / "models" / TASK_MODEL_DIRS[task] / "best.pt"


def main() -> None:
    """部署包统一推理入口。

    示例：
        python predict.py --task en_de_translation --text "a man is riding a bike" --device cpu
        python predict.py --task zh_sentiment --text "房间很干净，服务也很好。" --device cpu
    """
    parser = argparse.ArgumentParser(description="Unified inference entry for packaged Transformer models.")
    parser.add_argument(
        "--task",
        required=True,
        choices=sorted(TASK_MODEL_DIRS),
        help="任务名",
    )
    parser.add_argument("--text", required=True, help="待预测文本")
    parser.add_argument("--device", default="cpu", help="cpu/cuda/auto")
    parser.add_argument("--max-len", type=int, default=80, help="翻译最大生成长度")
    parser.add_argument("--package-root", default=str(ROOT), help="部署包根目录")
    args = parser.parse_args()

    package_root = Path(args.package_root)
    checkpoint = checkpoint_for(package_root, args.task)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if args.task in {"en_de_translation", "en_zh_translation"}:
        model, src_vocab, tgt_vocab, config, device = load_checkpoint(checkpoint, args.device)
        output = greedy_translate(
            model,
            src_vocab,
            tgt_vocab,
            args.text,
            device,
            max_len=args.max_len,
            lowercase=config.lowercase,
            src_char_level=config.src_char_level,
        )
        print(output)
        return

    if args.task == "ag_news":
        model, vocab, labels, config, device = load_classifier_checkpoint(checkpoint, args.device)
        label, prob, scores = classify_text(
            model,
            vocab,
            labels,
            args.text,
            device,
            max_len=config.max_len,
            lowercase=config.lowercase,
        )
    else:
        model, vocab, labels, config, device = load_chinese_sentiment_checkpoint(checkpoint, args.device)
        label, prob, scores = predict_chinese_sentiment(
            model,
            vocab,
            labels,
            args.text,
            device,
            max_len=config.max_len,
            lowercase=config.lowercase,
            char_level=config.char_level,
        )

    print(f"label={label} prob={prob:.4f}")
    for name, score in scores:
        print(f"{name}: {score:.4f}")


if __name__ == "__main__":
    main()
