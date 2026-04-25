from __future__ import annotations

import argparse
from pathlib import Path

import requests
from tqdm import tqdm


# AG News csv 镜像。文件格式为 label,title,description。
# 标签共有四类：World、Sports、Business、Sci/Tech。
URLS = {
    "train": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
    "test": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
}


def download(url: str, path: Path) -> None:
    """下载文件；如果已存在则跳过。"""
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
    """命令行入口：下载 AG News 到 dataset/ag_news。"""
    parser = argparse.ArgumentParser(description="Download AG News classification dataset.")
    parser.add_argument("--root", default="dataset/ag_news")
    args = parser.parse_args()

    root = Path(args.root)
    for split, url in URLS.items():
        download(url, root / f"{split}.csv")
    print(f"ready: {root}")


if __name__ == "__main__":
    main()
