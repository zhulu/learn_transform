from __future__ import annotations

import argparse
from pathlib import Path
import sys

# 让脚本可以从项目根目录直接运行，并正确导入 src。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference import greedy_translate, load_checkpoint


def main() -> None:
    """翻译入口脚本。

    加载 best.pt/last.pt checkpoint 后，对用户传入的一句话做贪心解码。
    """
    parser = argparse.ArgumentParser(description="Greedy decode with a trained Transformer checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/multi30k_en_de/best.pt")
    parser.add_argument("--text", required=True)
    parser.add_argument("--max-len", type=int, default=80)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    # checkpoint 同目录下需要有训练时保存的 src_vocab.json/tgt_vocab.json。
    model, src_vocab, tgt_vocab, config, device = load_checkpoint(args.checkpoint, args.device)
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


if __name__ == "__main__":
    main()
