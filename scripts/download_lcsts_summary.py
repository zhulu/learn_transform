from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

import requests
from tqdm import tqdm


# LCSTS 是中文短文本摘要数据集。这里使用 Hugging Face datasets-server 的 rows API，
# 避免额外引入 datasets/pyarrow 等较重依赖。字段含义：
# - text: 原文
# - summary: 中心思想/摘要/标题式概括
DATASET = "hugcyp/LCSTS"
CONFIG = "default"
API_ROOT = "https://datasets-server.huggingface.co"
MAX_PAGE_SIZE = 100


def get_split_sizes() -> dict[str, int]:
    """读取 LCSTS 每个 split 的样本数。"""
    url = f"{API_ROOT}/size"
    params = {"dataset": DATASET, "config": CONFIG}
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    return {
        item["split"]: int(item["num_rows"])
        for item in payload["size"]["splits"]
    }


def fetch_rows(split: str, limit: int, offset: int = 0) -> Iterator[dict[str, str]]:
    """分页拉取 rows API。

    datasets-server 单次最多返回 100 条，所以这里按页迭代。
    """
    fetched = 0
    progress = tqdm(total=limit, desc=f"fetch {split}")
    while fetched < limit:
        length = min(MAX_PAGE_SIZE, limit - fetched)
        url = f"{API_ROOT}/rows"
        params = {
            "dataset": DATASET,
            "config": CONFIG,
            "split": split,
            "offset": offset + fetched,
            "length": length,
        }
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        rows = response.json()["rows"]
        if not rows:
            break
        for item in rows:
            yield item["row"]
        fetched += len(rows)
        progress.update(len(rows))
        if len(rows) < length:
            break
    progress.close()


def normalize(text: str) -> str:
    """做非常轻量的文本清洗。

    LCSTS 是中文摘要数据，第一版用字符级 token 训练，所以这里去掉空白即可。
    """
    return "".join(text.strip().split())


def write_split(root: Path, name: str, rows: Iterator[dict[str, str]], keep_jsonl: bool) -> int:
    """写出一个 split。

    现有翻译训练入口要求文件名形如 split.src/split.tgt：
    - src: 原文
    - tgt: 中心思想/摘要
    """
    root.mkdir(parents=True, exist_ok=True)
    count = 0
    src_path = root / f"{name}.src"
    tgt_path = root / f"{name}.tgt"
    jsonl_path = root / f"{name}.jsonl"

    jsonl_file = jsonl_path.open("w", encoding="utf-8") if keep_jsonl else None
    try:
        with src_path.open("w", encoding="utf-8") as src_f, tgt_path.open("w", encoding="utf-8") as tgt_f:
            for row in rows:
                text = normalize(row.get("text", ""))
                summary = normalize(row.get("summary", ""))
                if not text or not summary:
                    continue
                src_f.write(text + "\n")
                tgt_f.write(summary + "\n")
                if jsonl_file is not None:
                    jsonl_file.write(json.dumps({"text": text, "summary": summary}, ensure_ascii=False) + "\n")
                count += 1
    finally:
        if jsonl_file is not None:
            jsonl_file.close()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Download LCSTS Chinese summarization data.")
    parser.add_argument("--root", default="dataset/lcsts_summary")
    parser.add_argument("--train-size", type=int, default=100000)
    parser.add_argument("--valid-size", type=int, default=4000)
    parser.add_argument("--test-size", type=int, default=4000)
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--keep-jsonl", action="store_true")
    args = parser.parse_args()

    split_sizes = get_split_sizes()
    train_size = min(args.train_size, split_sizes["train"] - args.train_offset)
    labeled_eval_size = split_sizes["validation"]
    valid_size = min(args.valid_size, labeled_eval_size)
    test_size = min(args.test_size, max(0, labeled_eval_size - valid_size))

    root = Path(args.root)
    train_count = write_split(
        root,
        "train",
        fetch_rows("train", train_size, offset=args.train_offset),
        keep_jsonl=args.keep_jsonl,
    )
    # Hugging Face 版本的 LCSTS test split 没有公开 summary 标签，不能直接作为
    # 有监督测试集。因此这里从有标签的 validation split 中切出 valid/test。
    eval_rows = list(fetch_rows("validation", valid_size + test_size))
    valid_count = write_split(
        root,
        "valid",
        iter(eval_rows[:valid_size]),
        keep_jsonl=args.keep_jsonl,
    )
    test_count = write_split(
        root,
        "test",
        iter(eval_rows[valid_size : valid_size + test_size]),
        keep_jsonl=args.keep_jsonl,
    )
    print(f"train={train_count} valid={valid_count} test={test_count} ready: {root}")


if __name__ == "__main__":
    main()
