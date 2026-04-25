from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .vocab import Vocabulary


# 一个简单的英文/西文分词正则：
# - [A-Za-z]+(?:'[A-Za-z]+)? 匹配英文单词和 don't 这类缩写。
# - \d+ 匹配数字。
# - [^\sA-Za-z\d] 匹配标点和其他单字符符号。
# 中文任务后面可以设置 char_level=True，先按字符切分，避免引入额外中文分词库。
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\sA-Za-z\d]", re.UNICODE)


def tokenize(text: str, lowercase: bool = True, char_level: bool = False) -> list[str]:
    """把原始句子切成 token。

    Args:
        text: 一行原始文本。
        lowercase: 是否转小写。Multi30k 这种入门数据集转小写可以减少词表规模。
        char_level: 是否按字符切分。英中翻译时，中文目标端可以先用字符级 token。

    Returns:
        list[str]: token 序列，例如 "A man." -> ["a", "man", "."]。
    """
    text = text.strip()
    if lowercase:
        text = text.lower()
    if char_level:
        return [ch for ch in text if not ch.isspace()]
    return _WORD_RE.findall(text)


def read_lines(path: str | Path) -> list[str]:
    """读取文本文件，返回去掉换行符后的每一行。"""
    with Path(path).open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def read_parallel(src_path: str | Path, tgt_path: str | Path) -> list[tuple[str, str]]:
    """读取平行语料。

    平行语料要求源语言文件和目标语言文件行数一致，且第 n 行互为翻译。
    例如：
    - train.en 第 10 行：a man is riding a bike .
    - train.de 第 10 行：ein mann fährt fahrrad .
    """
    src_lines = read_lines(src_path)
    tgt_lines = read_lines(tgt_path)
    if len(src_lines) != len(tgt_lines):
        raise ValueError(f"Line count mismatch: {src_path}={len(src_lines)}, {tgt_path}={len(tgt_lines)}")
    return list(zip(src_lines, tgt_lines))


@dataclass
class TranslationExample:
    """一个已经数值化的翻译样本。

    src: 源语言 id 序列，形如 [12, 35, 9, eos]。
    tgt: 目标语言 id 序列，形如 [bos, 44, 7, 18, eos]。
    """

    src: list[int]
    tgt: list[int]


class ParallelTextDataset(Dataset[TranslationExample]):
    """PyTorch Dataset：把平行文本文件转成模型可训练的样本。

    Dataset 的职责是“按 index 返回一个样本”，不负责 padding。
    padding 交给 collate_fn，因为只有拿到一个 batch 后，才知道这个 batch
    内最长句子是多少。
    """

    def __init__(
        self,
        src_path: str | Path,
        tgt_path: str | Path,
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        max_len: int = 128,
        lowercase: bool = True,
        src_char_level: bool = False,
        tgt_char_level: bool = False,
    ) -> None:
        self.examples: list[TranslationExample] = []
        for src_text, tgt_text in read_parallel(src_path, tgt_path):
            src_tokens = tokenize(src_text, lowercase=lowercase, char_level=src_char_level)
            tgt_tokens = tokenize(tgt_text, lowercase=lowercase, char_level=tgt_char_level)
            # 空行没有训练价值，直接跳过。
            if not src_tokens or not tgt_tokens:
                continue
            # 过长句子会显著增加显存占用；入门训练先过滤掉。
            if len(src_tokens) > max_len or len(tgt_tokens) > max_len:
                continue
            self.examples.append(
                TranslationExample(
                    # encoder 输入不需要 <bos>，但需要 <eos> 告诉模型源句结束。
                    src=src_vocab.encode(src_tokens, add_bos=False, add_eos=True),
                    # decoder 训练需要 <bos> 作为第一个输入，并以 <eos> 作为停止标签。
                    tgt=tgt_vocab.encode(tgt_tokens, add_bos=True, add_eos=True),
                )
            )

    def __len__(self) -> int:
        """返回过滤后的样本数量。"""
        return len(self.examples)

    def __getitem__(self, index: int) -> TranslationExample:
        """按下标返回一个样本，DataLoader 会反复调用它组 batch。"""
        return self.examples[index]


def build_vocabs_from_files(
    src_path: str | Path,
    tgt_path: str | Path,
    src_vocab_size: int = 16000,
    tgt_vocab_size: int = 16000,
    min_freq: int = 2,
    lowercase: bool = True,
    src_char_level: bool = False,
    tgt_char_level: bool = False,
) -> tuple[Vocabulary, Vocabulary]:
    """从训练集文本构建源语言和目标语言词表。

    验证集/测试集不参与建词表，这是机器学习里常见的数据隔离习惯。
    """
    pairs = read_parallel(src_path, tgt_path)
    src_tokens = (tokenize(src, lowercase=lowercase, char_level=src_char_level) for src, _ in pairs)
    tgt_tokens = (tokenize(tgt, lowercase=lowercase, char_level=tgt_char_level) for _, tgt in pairs)
    return (
        Vocabulary.build(src_tokens, max_size=src_vocab_size, min_freq=min_freq),
        Vocabulary.build(tgt_tokens, max_size=tgt_vocab_size, min_freq=min_freq),
    )


def make_collate_fn(src_pad_id: int, tgt_pad_id: int):
    """创建 DataLoader 使用的 batch 拼接函数。

    每个样本长度不同，不能直接堆成 Tensor。collate 会做两件事：
    1. 找出这个 batch 里源语言/目标语言的最大长度。
    2. 用 <pad> id 把短句补齐到最大长度。

    返回张量形状：
    - src: [batch_size, src_seq_len]
    - tgt: [batch_size, tgt_seq_len]
    """

    def collate(batch: list[TranslationExample]) -> tuple[torch.Tensor, torch.Tensor]:
        src_max = max(len(item.src) for item in batch)
        tgt_max = max(len(item.tgt) for item in batch)
        # 先创建全是 pad 的矩阵，再把真实 token id 填进去。
        src = torch.full((len(batch), src_max), src_pad_id, dtype=torch.long)
        tgt = torch.full((len(batch), tgt_max), tgt_pad_id, dtype=torch.long)
        for row, item in enumerate(batch):
            src[row, : len(item.src)] = torch.tensor(item.src, dtype=torch.long)
            tgt[row, : len(item.tgt)] = torch.tensor(item.tgt, dtype=torch.long)
        return src, tgt

    return collate
