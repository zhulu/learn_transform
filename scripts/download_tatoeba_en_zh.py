from __future__ import annotations

import argparse
import random
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


# OPUS Tatoeba 的 Mandarin Chinese-English Moses 格式下载地址。
# OPUS 中普通话中文使用语言代码 cmn；为了训练脚本更直观，整理后输出为 .zh。
URL = "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/cmn-en.txt.zip"


def download(url: str, path: Path) -> None:
    """下载 zip 文件；如果本地已存在则跳过。"""
    if path.exists() and path.stat().st_size > 0:
        print(f"exists: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with path.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=path.name) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def read_zip_lines(archive: Path) -> tuple[list[str], list[str]]:
    """从 OPUS zip 中读取英文和中文文件。

    zip 内部文件名通常是：
    - Tatoeba.cmn-en.en
    - Tatoeba.cmn-en.cmn
    """
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        en_name = next(name for name in names if name.endswith(".en"))
        zh_name = next(name for name in names if name.endswith(".cmn"))
        with z.open(en_name) as f:
            en_lines = [line.decode("utf-8").strip() for line in f]
        with z.open(zh_name) as f:
            zh_lines = [line.decode("utf-8").strip() for line in f]
    if len(en_lines) != len(zh_lines):
        raise ValueError(f"Line count mismatch: en={len(en_lines)}, zh={len(zh_lines)}")
    return en_lines, zh_lines


def normalize_pairs(
    en_lines: list[str],
    zh_lines: list[str],
    max_pairs: int,
    max_en_chars: int,
    max_zh_chars: int,
    seed: int,
) -> list[tuple[str, str]]:
    """清洗、去重并抽样英中句对。

    Tatoeba 里会有重复句对。这里用 set 去重，并过滤过长样本，让 3090 上的小模型
    可以更快开始训练。
    """
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []
    for en, zh in zip(en_lines, zh_lines):
        en = " ".join(en.split())
        zh = "".join(zh.split())
        if not en or not zh:
            continue
        if len(en) > max_en_chars or len(zh) > max_zh_chars:
            continue
        pair = (en, zh)
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)

    rng = random.Random(seed)
    rng.shuffle(pairs)
    if max_pairs > 0:
        pairs = pairs[:max_pairs]
    return pairs


def write_split(root: Path, split: str, pairs: list[tuple[str, str]]) -> None:
    """写出 train/valid/test.en 和 train/valid/test.zh。"""
    with (root / f"{split}.en").open("w", encoding="utf-8") as en_f, (
        root / f"{split}.zh"
    ).open("w", encoding="utf-8") as zh_f:
        for en, zh in pairs:
            en_f.write(en + "\n")
            zh_f.write(zh + "\n")


def main() -> None:
    """下载并整理 Tatoeba 英中小语料。"""
    parser = argparse.ArgumentParser(description="Download and split OPUS Tatoeba English-Chinese corpus.")
    parser.add_argument("--root", default="dataset/tatoeba_en_zh")
    parser.add_argument("--max-pairs", type=int, default=50000)
    parser.add_argument("--valid-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--max-en-chars", type=int, default=180)
    parser.add_argument("--max-zh-chars", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.root)
    raw = root / "raw"
    archive = raw / "tatoeba_cmn_en.zip"
    download(URL, archive)

    en_lines, zh_lines = read_zip_lines(archive)
    pairs = normalize_pairs(
        en_lines,
        zh_lines,
        max_pairs=args.max_pairs,
        max_en_chars=args.max_en_chars,
        max_zh_chars=args.max_zh_chars,
        seed=args.seed,
    )
    if len(pairs) <= args.valid_size + args.test_size:
        raise ValueError("Not enough pairs after filtering. Lower valid/test size or increase max-pairs.")

    valid = pairs[: args.valid_size]
    test = pairs[args.valid_size : args.valid_size + args.test_size]
    train = pairs[args.valid_size + args.test_size :]
    root.mkdir(parents=True, exist_ok=True)
    write_split(root, "train", train)
    write_split(root, "valid", valid)
    write_split(root, "test", test)
    print(f"train={len(train)} valid={len(valid)} test={len(test)} ready: {root}")


if __name__ == "__main__":
    main()
