from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm


# Multi30k 的小型镜像地址。
# 这个脚本会下载压缩包，再整理成统一的 train.en/train.de 等文件名。
URLS = {
    "train": "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz",
    "valid": "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz",
    "test": "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt16_task1_test.tar.gz",
}


def download(url: str, path: Path) -> None:
    """下载一个文件到本地。

    如果文件已经存在且大小大于 0，就认为已经下载过，直接跳过。
    这让脚本可以反复运行，不会每次都重新下载。
    """
    if path.exists() and path.stat().st_size > 0:
        print(f"exists: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with path.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=path.name) as bar:
            # 分块写入，避免把整个压缩包一次性读进内存。
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def extract(archive: Path, out_dir: Path) -> None:
    """解压 tar.gz 文件。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        # filter="data" 是 Python 新版推荐的安全解压方式。
        tar.extractall(out_dir, filter="data")


def find_file(root: Path, names: list[str]) -> Path:
    """在解压目录中查找候选文件名。

    不同 split 的原始文件命名略有差异，比如验证集叫 val.en，
    所以这里允许传入多个候选名。
    """
    for name in names:
        matches = list(root.rglob(name))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Cannot find any of {names} under {root}")


def copy_text(src: Path, dst: Path) -> None:
    """用统一编码复制文本文件。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = src.read_text(encoding="utf-8")
    dst.write_text(text, encoding="utf-8")


def main() -> None:
    """命令行入口：下载、解压、整理 Multi30k。"""
    parser = argparse.ArgumentParser(description="Download and normalize Multi30k en-de files.")
    parser.add_argument("--root", default="dataset/multi30k")
    args = parser.parse_args()

    root = Path(args.root)
    raw = root / "raw"
    extracted = root / "extracted"

    # 先下载每个 split 的压缩包，再解压到 extracted/{split}。
    for split, url in URLS.items():
        archive = raw / f"{split}.tar.gz"
        download(url, archive)
        extract(archive, extracted / split)

    # 整理成训练脚本默认识别的统一文件名。
    mapping = {
        "train": ("train.en", "train.de"),
        "valid": ("val.en", "val.de"),
        "test": ("test.en", "test.de"),
    }
    for split, (en_name, de_name) in mapping.items():
        split_root = extracted / split
        en = find_file(split_root, [en_name, f"{split}.en", "test2016.en"])
        de = find_file(split_root, [de_name, f"{split}.de", "test2016.de"])
        copy_text(en, root / f"{split}.en")
        copy_text(de, root / f"{split}.de")
        print(f"{split}: {en.name} -> {split}.en, {de.name} -> {split}.de")

    print(f"ready: {root}")


if __name__ == "__main__":
    main()
