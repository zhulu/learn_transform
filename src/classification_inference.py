from __future__ import annotations

from pathlib import Path

import torch

from .classification_data import AG_NEWS_LABELS
from .classification_training import ClassificationConfig, create_classifier
from .data import tokenize
from .training import resolve_device
from .vocab import Vocabulary


def load_classifier_checkpoint(checkpoint_path: str | Path, device_name: str = "auto"):
    """加载文本分类 checkpoint。"""
    checkpoint_path = Path(checkpoint_path)
    device = resolve_device(device_name)
    payload = torch.load(checkpoint_path, map_location=device)
    save_dir = checkpoint_path.parent
    vocab = Vocabulary.load(save_dir / "vocab.json")
    config = ClassificationConfig(**payload["config"])
    labels = payload.get("labels", AG_NEWS_LABELS)
    model = create_classifier(config, vocab)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model, vocab, labels, config, device


@torch.no_grad()
def classify_text(
    model,
    vocab: Vocabulary,
    labels: list[str],
    text: str,
    device: torch.device,
    max_len: int = 128,
    lowercase: bool = True,
) -> tuple[str, float, list[tuple[str, float]]]:
    """对单条文本做分类。

    返回：
    - 最可能类别名。
    - 最可能类别概率。
    - 所有类别及其概率，便于观察模型信心。
    """
    tokens = tokenize(text, lowercase=lowercase)[:max_len]
    ids = vocab.encode(tokens, add_bos=False, add_eos=True)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(input_ids)
    probs = torch.softmax(logits, dim=-1).squeeze(0)
    best_id = int(probs.argmax().item())
    all_scores = [(label, float(probs[idx].item())) for idx, label in enumerate(labels)]
    return labels[best_id], float(probs[best_id].item()), all_scores
