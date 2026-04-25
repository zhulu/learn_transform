from __future__ import annotations

import argparse
from pathlib import Path
import sys

# 脚本从 scripts/ 目录运行时，Python 默认找不到项目根目录下的 src。
# 这里把项目根目录插到 sys.path 最前面，保证可以 import src.training。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training import TrainConfig, train_translation


def main() -> None:
    """训练入口脚本。

    它只负责解析命令行参数，并组装 TrainConfig。
    真正的训练逻辑放在 src/training.py，方便后面被其他脚本复用。
    """
    parser = argparse.ArgumentParser(description="Train a small Transformer translation model.")
    # 数据文件按 train.{src_lang}/train.{tgt_lang}/valid.{src_lang}/valid.{tgt_lang} 命名。
    parser.add_argument("--data-root", default="dataset/multi30k")
    parser.add_argument("--src-lang", default="en")
    parser.add_argument("--tgt-lang", default="de")
    parser.add_argument("--save-dir", default="checkpoints/multi30k_en_de")
    # 训练超参数。
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=128)
    # 模型结构超参数。
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    # 词表和数据加载参数。
    parser.add_argument("--src-vocab-size", type=int, default=16000)
    parser.add_argument("--tgt-vocab-size", type=int, default=16000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--src-char-level", action="store_true")
    parser.add_argument("--tgt-char-level", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    # 根据 data-root 和语言后缀拼出平行语料路径。
    config = TrainConfig(
        train_src=str(data_root / f"train.{args.src_lang}"),
        train_tgt=str(data_root / f"train.{args.tgt_lang}"),
        valid_src=str(data_root / f"valid.{args.src_lang}"),
        valid_tgt=str(data_root / f"valid.{args.tgt_lang}"),
        save_dir=args.save_dir,
        src_vocab_size=args.src_vocab_size,
        tgt_vocab_size=args.tgt_vocab_size,
        min_freq=args.min_freq,
        max_len=args.max_len,
        d_model=args.d_model,
        num_heads=args.heads,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_workers=args.num_workers,
        device=args.device,
        amp=not args.no_amp,
        src_char_level=args.src_char_level,
        tgt_char_level=args.tgt_char_level,
    )
    # 进入 src/training.py 中的完整训练流程。
    train_translation(config)


if __name__ == "__main__":
    main()
