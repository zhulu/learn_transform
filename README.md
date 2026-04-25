# learn_transform

Minimal Transformer machine translation project built mostly from scratch with PyTorch.

## Layout

- `src/data.py`: parallel text loading, tokenization, batching
- `src/vocab.py`: simple vocabulary encode/decode
- `src/model.py`: Transformer encoder-decoder, attention, masks, positional encoding
- `src/training.py`: train/eval loop and checkpoint saving
- `src/inference.py`: checkpoint loading and greedy decoding
- `scripts/download_multi30k.py`: download Multi30k into `dataset/multi30k`
- `scripts/train_translation.py`: train entry point
- `scripts/translate.py`: inference entry point

## Prepare Data

```powershell
python scripts\download_multi30k.py --root dataset\multi30k
```

This creates:

```text
dataset/multi30k/train.en
dataset/multi30k/train.de
dataset/multi30k/valid.en
dataset/multi30k/valid.de
dataset/multi30k/test.en
dataset/multi30k/test.de
```

## Train On A 3090

Small and friendly first run:

```powershell
python scripts\train_translation.py `
  --data-root dataset\multi30k `
  --src-lang en `
  --tgt-lang de `
  --save-dir checkpoints\multi30k_en_de `
  --epochs 20 `
  --batch-size 128 `
  --max-len 128 `
  --d-model 256 `
  --heads 4 `
  --encoder-layers 3 `
  --decoder-layers 3 `
  --d-ff 1024
```

Larger but still reasonable on a 24GB 3090:

```powershell
python scripts\train_translation.py `
  --data-root dataset\multi30k `
  --src-lang en `
  --tgt-lang de `
  --save-dir checkpoints\multi30k_en_de_512 `
  --epochs 30 `
  --batch-size 96 `
  --max-len 128 `
  --d-model 512 `
  --heads 8 `
  --encoder-layers 4 `
  --decoder-layers 4 `
  --d-ff 2048
```

## Translate

```powershell
python scripts\translate.py `
  --checkpoint checkpoints\multi30k_en_de\best.pt `
  --text "a man is riding a bike"
```

## Note

The current dataset path is English to German because Multi30k is the quickest way to validate a from-scratch Transformer. The data interface is language-agnostic, so English to Chinese can use the same training script once `train.en`, `train.zh`, `valid.en`, and `valid.zh` are prepared under a dataset folder.
