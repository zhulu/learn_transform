from __future__ import annotations

import argparse
from pathlib import Path

import requests
from tqdm import tqdm


# ChnSentiCorp 是中文情感分类常用小数据集，格式为 label<TAB>text_a。
URLS = {
    "train": "https://raw.githubusercontent.com/duanruixue/chnsenticorp/main/train.tsv",
    "valid": "https://raw.githubusercontent.com/duanruixue/chnsenticorp/main/dev.tsv",
    "test": "https://raw.githubusercontent.com/duanruixue/chnsenticorp/main/test.tsv",
}


def download(url: str, path: Path) -> None:
    """下载一个 tsv 文件；如果已存在则跳过。"""
    if path.exists() and path.stat().st_size > 0:
        print(f"exists: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with path.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=path.name) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def main() -> None:
    """命令行入口：下载 ChnSentiCorp 到 dataset/chnsenticorp。"""
    parser = argparse.ArgumentParser(description="Download ChnSentiCorp Chinese sentiment dataset.")
    parser.add_argument("--root", default="dataset/chnsenticorp")
    args = parser.parse_args()

    root = Path(args.root)
    for split, url in URLS.items():
        download(url, root / f"{split}.tsv")
    print(f"ready: {root}")


if __name__ == "__main__":
    main()
