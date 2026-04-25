from __future__ import annotations

import math

import torch
from torch import nn

from .model import EncoderLayer, PositionalEncoding


class TransformerTextClassifier(nn.Module):
    """基于 Transformer Encoder 的文本分类模型。

    翻译任务使用 encoder-decoder；分类任务只需要理解输入文本，所以只用 encoder。

    整体流程：
    1. token id -> embedding。
    2. 加位置编码。
    3. 经过多层 Transformer encoder。
    4. 对非 pad token 做平均池化，得到整句向量。
    5. 线性分类层输出每个类别的 logits。

    输入：
    - input_ids: [batch_size, seq_len]

    输出：
    - logits: [batch_size, num_classes]
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        num_classes: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.positional = PositionalEncoding(d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self._reset_parameters()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """前向传播。

        input_ids 中的 <pad> 会被 mask 掉，不参与 attention 和平均池化。
        """
        mask = self.make_mask(input_ids)
        x = self.dropout(self.positional(self.embedding(input_ids) * math.sqrt(self.d_model)))
        for layer in self.encoder:
            x = layer(x, mask)
        x = self.norm(x)
        pooled = self.mean_pool(x, input_ids)
        return self.classifier(pooled)

    def make_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """构造 encoder padding mask。

        输入 input_ids: [B, L]
        输出 mask: [B, 1, 1, L]

        这个形状和翻译 encoder 的 src_mask 一致，可以复用 EncoderLayer。
        """
        return (input_ids != self.pad_id).unsqueeze(1).unsqueeze(2)

    def mean_pool(self, hidden: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """对非 pad token 的 hidden state 做平均池化。

        hidden: [B, L, D]
        返回: [B, D]

        这里不用额外引入 [CLS] token，是为了保持词表和数据处理更简单。
        """
        valid = (input_ids != self.pad_id).unsqueeze(-1)
        hidden = hidden * valid
        lengths = valid.sum(dim=1).clamp(min=1)
        return hidden.sum(dim=1) / lengths

    def _reset_parameters(self) -> None:
        """初始化矩阵参数。"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
