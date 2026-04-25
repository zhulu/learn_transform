from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .data import tokenize
from .vocab import Vocabulary


CHNSENTICORP_LABELS = ["negative", "positive"]


@dataclass
class ChineseSentimentExample:
    """一个中文情感分类样本。"""

    text: list[int]
    label: int


def read_chnsenticorp_tsv(path: str | Path) -> list[tuple[str, int]]:
    """读取 ChnSentiCorp tsv。

    文件包含表头：
        label    text_a

    label 为 0/1，分别表示负向/正向。
    """
    rows: list[tuple[str, int]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = (row.get("text_a") or "").strip()
            label_text = (row.get("label") or "").strip()
            if not text or label_text not in {"0", "1"}:
                continue
            rows.append((text, int(label_text)))
    return rows


def build_vocab_from_chnsenticorp(
    train_tsv: str | Path,
    vocab_size: int = 8000,
    min_freq: int = 2,
    lowercase: bool = True,
    char_level: bool = True,
) -> Vocabulary:
    """从中文情感训练集构建词表。

    中文第一版默认使用字符级 token，避免额外安装 jieba 等分词库。
    """
    rows = read_chnsenticorp_tsv(train_tsv)
    tokenized = (tokenize(text, lowercase=lowercase, char_level=char_level) for text, _ in rows)
    return Vocabulary.build(tokenized, max_size=vocab_size, min_freq=min_freq)


class ChineseSentimentDataset(Dataset[ChineseSentimentExample]):
    """ChnSentiCorp Dataset。"""

    def __init__(
        self,
        tsv_path: str | Path,
        vocab: Vocabulary,
        max_len: int = 256,
        lowercase: bool = True,
        char_level: bool = True,
    ) -> None:
        self.examples: list[ChineseSentimentExample] = []
        for text, label in read_chnsenticorp_tsv(tsv_path):
            tokens = tokenize(text, lowercase=lowercase, char_level=char_level)
            if not tokens:
                continue
            tokens = tokens[:max_len]
            ids = vocab.encode(tokens, add_bos=False, add_eos=True)
            self.examples.append(ChineseSentimentExample(text=ids, label=label))

    def __len__(self) -> int:
        """返回样本数。"""
        return len(self.examples)

    def __getitem__(self, index: int) -> ChineseSentimentExample:
        """返回一个样本。"""
        return self.examples[index]


def make_chinese_sentiment_collate_fn(pad_id: int):
    """创建中文情感分类的 batch 拼接函数。"""

    def collate(batch: list[ChineseSentimentExample]) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(len(item.text) for item in batch)
        text = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        label = torch.empty(len(batch), dtype=torch.long)
        for row, item in enumerate(batch):
            text[row, : len(item.text)] = torch.tensor(item.text, dtype=torch.long)
            label[row] = item.label
        return text, label

    return collate
