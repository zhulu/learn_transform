from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .data import tokenize
from .vocab import Vocabulary


# AG News 原始 csv 中的标签是 1..4，这里转成训练更常用的 0..3。
# 模型输出 logits 的第 0/1/2/3 维会分别对应这些类别。
AG_NEWS_LABELS = ["World", "Sports", "Business", "Sci/Tech"]


@dataclass
class ClassificationExample:
    """一个文本分类样本。

    text: 已经数值化后的 token id 序列。
    label: 类别 id，从 0 开始。
    """

    text: list[int]
    label: int


def read_ag_news_csv(path: str | Path) -> list[tuple[str, int]]:
    """读取 AG News csv 文件。

    原始文件每行格式为：
        label,title,description

    其中 label 是 1..4。我们会把 title 和 description 拼成一个完整文本，
    并把 label 减 1，得到 0..3 的类别 id。
    """
    rows: list[tuple[str, int]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            label = int(row[0]) - 1
            text = f"{row[1]} {row[2]}".strip()
            rows.append((text, label))
    return rows


def build_vocab_from_classification_file(
    train_csv: str | Path,
    vocab_size: int = 30000,
    min_freq: int = 2,
    lowercase: bool = True,
) -> Vocabulary:
    """从文本分类训练集构建词表。

    仍然只用训练集建词表，测试集不参与，避免信息泄漏。
    """
    rows = read_ag_news_csv(train_csv)
    tokenized = (tokenize(text, lowercase=lowercase) for text, _ in rows)
    return Vocabulary.build(tokenized, max_size=vocab_size, min_freq=min_freq)


class TextClassificationDataset(Dataset[ClassificationExample]):
    """文本分类 Dataset。

    它负责把 csv 里的原始文本转成 token id，并过滤过长或空文本。
    padding 仍然交给 collate_fn 在 batch 级别处理。
    """

    def __init__(
        self,
        csv_path: str | Path,
        vocab: Vocabulary,
        max_len: int = 128,
        lowercase: bool = True,
    ) -> None:
        self.examples: list[ClassificationExample] = []
        for text, label in read_ag_news_csv(csv_path):
            tokens = tokenize(text, lowercase=lowercase)
            if not tokens:
                continue
            # 分类任务中通常保留句首信息就够了；超过 max_len 的文本直接截断。
            tokens = tokens[:max_len]
            ids = vocab.encode(tokens, add_bos=False, add_eos=True)
            self.examples.append(ClassificationExample(text=ids, label=label))

    def __len__(self) -> int:
        """返回样本数量。"""
        return len(self.examples)

    def __getitem__(self, index: int) -> ClassificationExample:
        """按下标返回一个分类样本。"""
        return self.examples[index]


def make_classification_collate_fn(pad_id: int):
    """创建文本分类任务的 batch 拼接函数。

    返回：
    - text: [batch_size, seq_len]，padding 后的 token id。
    - label: [batch_size]，类别 id。
    """

    def collate(batch: list[ClassificationExample]) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(len(item.text) for item in batch)
        text = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        label = torch.empty(len(batch), dtype=torch.long)
        for row, item in enumerate(batch):
            text[row, : len(item.text)] = torch.tensor(item.text, dtype=torch.long)
            label[row] = item.label
        return text, label

    return collate
