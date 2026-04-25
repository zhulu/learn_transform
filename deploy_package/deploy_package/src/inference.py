from __future__ import annotations

from pathlib import Path

import torch

from .data import tokenize
from .model import Transformer
from .training import TrainConfig, create_model, resolve_device
from .vocab import Vocabulary


def load_checkpoint(checkpoint_path: str | Path, device_name: str = "auto"):
    """加载训练好的 checkpoint、词表和配置。

    checkpoint 里保存的是模型参数 state_dict 和训练配置；
    词表单独保存在 checkpoint 同目录下的 src_vocab.json/tgt_vocab.json。
    """
    checkpoint_path = Path(checkpoint_path)
    device = resolve_device(device_name)
    payload = torch.load(checkpoint_path, map_location=device)
    save_dir = checkpoint_path.parent
    # 推理时必须使用训练时同一份词表，否则 token id 的含义会错位。
    src_vocab = Vocabulary.load(save_dir / "src_vocab.json")
    tgt_vocab = Vocabulary.load(save_dir / "tgt_vocab.json")
    config = TrainConfig(**payload["config"])
    model = create_model(config, src_vocab, tgt_vocab)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model, src_vocab, tgt_vocab, config, device


@torch.no_grad()
def greedy_translate(
    model: Transformer,
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    text: str,
    device: torch.device,
    max_len: int = 80,
    lowercase: bool = True,
    src_char_level: bool = False,
) -> str:
    """使用贪心搜索做翻译。

    贪心搜索每一步都选择概率最高的下一个 token：
    1. 先把源句编码成 memory。
    2. 从 <bos> 开始生成目标句。
    3. 每次把当前已生成序列送进 decoder。
    4. 取最后一个位置的 logits，argmax 得到下一个 token。
    5. 如果生成 <eos> 或达到 max_len 就停止。

    这是最简单的解码方式，后面可以升级成 beam search。
    """
    tokens = tokenize(text, lowercase=lowercase, char_level=src_char_level)
    src_ids = src_vocab.encode(tokens, add_bos=False, add_eos=True)
    # 单句推理也要保留 batch 维度，所以形状是 [1, src_len]。
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask = model.make_src_mask(src)
    # 源句只需要编码一次，后续每一步生成都复用 memory。
    memory = model.encode(src, src_mask)

    # decoder 从 <bos> 开始。
    ys = torch.tensor([[tgt_vocab.bos_id]], dtype=torch.long, device=device)
    for _ in range(max_len):
        tgt_mask = model.make_tgt_mask(ys)
        decoded = model.decode(ys, memory, tgt_mask, src_mask)
        # decoded[:, -1] 是当前最后一个位置的隐状态，用它预测下一个 token。
        logits = model.generator(decoded[:, -1])
        next_id = int(logits.argmax(dim=-1).item())
        ys = torch.cat([ys, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
        if next_id == tgt_vocab.eos_id:
            break
    return " ".join(tgt_vocab.decode(ys.squeeze(0).tolist()))
