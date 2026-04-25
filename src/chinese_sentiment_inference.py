from __future__ import annotations

from pathlib import Path

import torch

from .chinese_sentiment_data import CHNSENTICORP_LABELS
from .chinese_sentiment_training import ChineseSentimentConfig, create_chinese_sentiment_model
from .data import tokenize
from .training import resolve_device
from .vocab import Vocabulary


def load_chinese_sentiment_checkpoint(checkpoint_path: str | Path, device_name: str = "auto"):
    """加载中文情感分类 checkpoint。"""
    checkpoint_path = Path(checkpoint_path)
    device = resolve_device(device_name)
    payload = torch.load(checkpoint_path, map_location=device)
    save_dir = checkpoint_path.parent
    vocab = Vocabulary.load(save_dir / "vocab.json")
    config = ChineseSentimentConfig(**payload["config"])
    labels = payload.get("labels", CHNSENTICORP_LABELS)
    model = create_chinese_sentiment_model(config, vocab)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model, vocab, labels, config, device


@torch.no_grad()
def predict_chinese_sentiment(
    model,
    vocab: Vocabulary,
    labels: list[str],
    text: str,
    device: torch.device,
    max_len: int = 256,
    lowercase: bool = True,
    char_level: bool = True,
) -> tuple[str, float, list[tuple[str, float]]]:
    """对单条中文文本预测情感类别。"""
    tokens = tokenize(text, lowercase=lowercase, char_level=char_level)[:max_len]
    ids = vocab.encode(tokens, add_bos=False, add_eos=True)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(input_ids)
    probs = torch.softmax(logits, dim=-1).squeeze(0)
    best_id = int(probs.argmax().item())
    all_scores = [(label, float(probs[idx].item())) for idx, label in enumerate(labels)]
    return labels[best_id], float(probs[best_id].item()), all_scores
