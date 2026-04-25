from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# 四个特殊 token 是序列模型里最常用的控制符号：
# - <pad>：补齐 batch 内不同长度的句子，loss 和 attention 都会忽略它。
# - <bos>：begin of sentence，解码器输入的第一个 token。
# - <eos>：end of sentence，模型生成到这里就可以停止。
# - <unk>：unknown，词表外 token 的兜底表示。
PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]


@dataclass
class Vocabulary:
    """一个非常轻量的词表类。

    Transformer 本身只能处理整数 id，不能直接处理字符串 token。
    这个类负责在 token 和 id 之间来回转换：
    - token_to_id: 从字符串 token 找到整数 id。
    - id_to_token: 从整数 id 找回字符串 token。

    这里没有使用 sentencepiece/BPE 等成熟 tokenizer，是为了让第一版工程
    尽量透明，方便观察“文本 -> token -> id -> 模型”的完整路径。
    """

    token_to_id: dict[str, int]
    id_to_token: list[str]

    @classmethod
    def build(
        cls,
        tokenized_lines: Iterable[list[str]],
        max_size: int = 16000,
        min_freq: int = 2,
    ) -> "Vocabulary":
        """从已经分词好的语料中统计词频并建立词表。

        Args:
            tokenized_lines: 每个元素是一句话的 token 列表，例如 ["a", "man", "."]。
            max_size: 词表最大容量，包含特殊 token。
            min_freq: token 至少出现多少次才进入词表。

        Returns:
            Vocabulary: 构造好的词表对象。

        词表构造策略很朴素：
        1. 统计所有 token 的出现次数。
        2. 按频率从高到低排序。
        3. 过滤低频词和特殊 token。
        4. 把特殊 token 放在词表最前面，保证 id 稳定。
        """
        counter: Counter[str] = Counter()
        for tokens in tokenized_lines:
            counter.update(tokens)

        # most_common 会返回 [(token, freq), ...]，天然按频率降序排列。
        words = [
            token
            for token, freq in counter.most_common()
            if freq >= min_freq and token not in SPECIAL_TOKENS
        ]
        # max_size 要包含特殊 token，所以普通词最多只能占剩余位置。
        words = words[: max(0, max_size - len(SPECIAL_TOKENS))]
        id_to_token = SPECIAL_TOKENS + words
        token_to_id = {token: idx for idx, token in enumerate(id_to_token)}
        return cls(token_to_id=token_to_id, id_to_token=id_to_token)

    @property
    def pad_id(self) -> int:
        """padding token 的 id，用于 batch 补齐和 loss 忽略。"""
        return self.token_to_id[PAD]

    @property
    def bos_id(self) -> int:
        """句子起始 token 的 id，解码器自回归生成从它开始。"""
        return self.token_to_id[BOS]

    @property
    def eos_id(self) -> int:
        """句子结束 token 的 id，推理时生成到它就停止。"""
        return self.token_to_id[EOS]

    @property
    def unk_id(self) -> int:
        """词表外 token 的 id。"""
        return self.token_to_id[UNK]

    def __len__(self) -> int:
        """返回词表大小，也就是 embedding 和输出分类层的类别数。"""
        return len(self.id_to_token)

    def encode(self, tokens: list[str], add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """把 token 列表转成 id 列表。

        Args:
            tokens: 已经分词后的 token。
            add_bos: 是否在开头添加 <bos>。
            add_eos: 是否在结尾添加 <eos>。

        源语言通常只需要在末尾加 <eos>；目标语言在训练时会加
        <bos> 和 <eos>，方便构造 decoder 输入/标签。
        """
        ids = [self.token_to_id.get(token, self.unk_id) for token in tokens]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> list[str]:
        """把 id 序列还原成 token 序列。

        Args:
            ids: 模型输出或数据集中保存的整数 id。
            skip_special: 是否跳过 <pad>/<bos>/<eos>/<unk> 这些特殊 token。

        注意：这里返回的是 token 列表，不负责把英文标点重新贴回单词。
        第一版保持简单，后面可以再加 detokenize。
        """
        tokens: list[str] = []
        for idx in ids:
            # 如果 id 越界，说明 checkpoint 或词表不匹配，用 <unk> 兜底。
            if idx < 0 or idx >= len(self.id_to_token):
                token = UNK
            else:
                token = self.id_to_token[idx]
            if skip_special and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return tokens

    def save(self, path: str | Path) -> None:
        """把词表保存成 json。

        只保存 id_to_token 就够了，因为 token_to_id 可以从它反推出来。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump({"id_to_token": self.id_to_token}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        """从 json 文件恢复词表。"""
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        id_to_token = list(payload["id_to_token"])
        return cls(
            token_to_id={token: idx for idx, token in enumerate(id_to_token)},
            id_to_token=id_to_token,
        )
