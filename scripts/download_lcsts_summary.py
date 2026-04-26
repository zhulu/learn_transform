from __future__ import annotations

import argparse
import json
import random
import time
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


def request_json(url: str, params: dict, max_retries: int, base_sleep: float) -> dict:
    """请求 Hugging Face datasets-server，并处理 429 限流。

    rows API 单页最多 100 条，下载 10 万条需要很多次请求。服务端偶尔会返回
    429 Too Many Requests，所以这里做指数退避重试。
    """
    for attempt in range(max_retries + 1):
        response = requests.get(url, params=params, timeout=60)
        if response.status_code != 429:
            response.raise_for_status()
            return response.json()

        retry_after = response.headers.get("retry-after")
        if retry_after is not None and retry_after.isdigit():
            sleep_seconds = float(retry_after)
        else:
            sleep_seconds = min(60.0, base_sleep * (2 ** attempt)) + random.uniform(0.0, 1.0)
        print(f"429 Too Many Requests, sleep {sleep_seconds:.1f}s then retry...")
        time.sleep(sleep_seconds)

    response.raise_for_status()
    raise RuntimeError("unreachable")


def get_split_sizes() -> dict[str, int]:
    """读取 LCSTS 每个 split 的样本数。"""
    url = f"{API_ROOT}/size"
    params = {"dataset": DATASET, "config": CONFIG}
    payload = request_json(url, params, max_retries=6, base_sleep=2.0)
    return {
        item["split"]: int(item["num_rows"])
        for item in payload["size"]["splits"]
    }


def fetch_rows(
    split: str,
    limit: int,
    offset: int = 0,
    max_retries: int = 8,
    request_sleep: float = 0.2,
) -> Iterator[dict[str, str]]:
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
        payload = request_json(url, params, max_retries=max_retries, base_sleep=max(1.0, request_sleep * 8))
        rows = payload["rows"]
        if not rows:
            break
        for item in rows:
            yield item["row"]
        fetched += len(rows)
        progress.update(len(rows))
        if request_sleep > 0:
            time.sleep(request_sleep)
        if len(rows) < length:
            break
    progress.close()


def normalize(text: str) -> str:
    """做非常轻量的文本清洗。

    LCSTS 是中文摘要数据，第一版用字符级 token 训练，所以这里去掉空白即可。
    """
    return "".join(text.strip().split())


def count_lines(path: Path) -> int:
    """统计文件行数。文件不存在时返回 0。"""
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def existing_pair_count(root: Path, name: str, keep_jsonl: bool) -> int:
    """返回一个 split 已经完整写好的样本数。"""
    src_count = count_lines(root / f"{name}.src")
    tgt_count = count_lines(root / f"{name}.tgt")
    if src_count != tgt_count:
        raise ValueError(f"{name}.src and {name}.tgt line counts mismatch. Use --overwrite to rebuild.")
    if keep_jsonl:
        jsonl_count = count_lines(root / f"{name}.jsonl")
        if jsonl_count not in {0, src_count}:
            raise ValueError(f"{name}.jsonl line count mismatch. Use --overwrite to rebuild.")
    return src_count


def remove_split(root: Path, name: str) -> None:
    """删除一个 split 的输出文件。"""
    for suffix in ("src", "tgt", "jsonl"):
        path = root / f"{name}.{suffix}"
        if path.exists():
            path.unlink()


def write_split(
    root: Path,
    name: str,
    rows: Iterator[dict[str, str]],
    keep_jsonl: bool,
    append: bool = False,
) -> int:
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

    mode = "a" if append else "w"
    jsonl_file = jsonl_path.open(mode, encoding="utf-8") if keep_jsonl else None
    try:
        with src_path.open(mode, encoding="utf-8") as src_f, tgt_path.open(mode, encoding="utf-8") as tgt_f:
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
    parser.add_argument("--overwrite", action="store_true", help="删除已有输出并重新下载")
    parser.add_argument("--request-sleep", type=float, default=0.3, help="每页请求后的等待秒数，降低 429 风险")
    parser.add_argument("--max-retries", type=int, default=10)
    args = parser.parse_args()

    split_sizes = get_split_sizes()
    train_size = min(args.train_size, split_sizes["train"] - args.train_offset)
    labeled_eval_size = split_sizes["validation"]
    valid_size = min(args.valid_size, labeled_eval_size)
    test_size = min(args.test_size, max(0, labeled_eval_size - valid_size))

    root = Path(args.root)
    if args.overwrite:
        for split in ("train", "valid", "test"):
            remove_split(root, split)

    train_existing = existing_pair_count(root, "train", args.keep_jsonl)
    if train_existing > train_size:
        raise ValueError("Existing train split is larger than requested. Use --overwrite to rebuild.")
    if train_existing:
        print(f"resume train from {train_existing}/{train_size}")
    train_count = train_existing + write_split(
        root,
        "train",
        fetch_rows(
            "train",
            train_size - train_existing,
            offset=args.train_offset + train_existing,
            max_retries=args.max_retries,
            request_sleep=args.request_sleep,
        ),
        keep_jsonl=args.keep_jsonl,
        append=train_existing > 0,
    )
    # Hugging Face 版本的 LCSTS test split 没有公开 summary 标签，不能直接作为
    # 有监督测试集。因此这里从有标签的 validation split 中切出 valid/test。
    valid_existing = existing_pair_count(root, "valid", args.keep_jsonl)
    test_existing = existing_pair_count(root, "test", args.keep_jsonl)
    if valid_existing > valid_size or test_existing > test_size:
        raise ValueError("Existing valid/test split is larger than requested. Use --overwrite to rebuild.")

    if valid_existing < valid_size or test_existing < test_size:
        # validation split 总量不大，直接拉取 valid+test 所需范围，再按已有行数补写。
        eval_rows = list(
            fetch_rows(
                "validation",
                valid_size + test_size,
                max_retries=args.max_retries,
                request_sleep=args.request_sleep,
            )
        )
        valid_count = valid_existing + write_split(
            root,
            "valid",
            iter(eval_rows[valid_existing:valid_size]),
            keep_jsonl=args.keep_jsonl,
            append=valid_existing > 0,
        )
        test_start = valid_size + test_existing
        test_count = test_existing + write_split(
            root,
            "test",
            iter(eval_rows[test_start : valid_size + test_size]),
            keep_jsonl=args.keep_jsonl,
            append=test_existing > 0,
        )
    else:
        valid_count = valid_existing
        test_count = test_existing
    print(f"train={train_count} valid={valid_count} test={test_count} ready: {root}")


if __name__ == "__main__":
    main()
