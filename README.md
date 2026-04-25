# learn_transform

这是一个从零开始学习 Transformer 的小工程。代码尽量只依赖 PyTorch，把词表、数据读取、mask、模型结构、训练循环、测试评估和推理流程都显式写出来。

当前包含四个任务，默认参数都按 RTX/ATX 3090 24G 显存做了保守设计：

1. **英德翻译**：Multi30k，encoder-decoder Transformer。
2. **英中翻译**：OPUS Tatoeba Mandarin Chinese-English 小语料，中文端默认字符级 token。
3. **英文新闻分类**：AG News，encoder-only Transformer。
4. **中文情感分类**：ChnSentiCorp，中文评论二分类，默认字符级 token。

## 环境准备

```bash
pip install -r requirements.txt
```

检查 CUDA 是否可用：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

如果输出 `True` 和显卡名称，训练脚本的 `--device auto` 会自动使用 GPU。

## 项目结构

```text
src/
  vocab.py                         词表构建、token/id 转换
  data.py                          翻译数据读取、分词、batch padding
  model.py                         手写 Transformer encoder-decoder
  training.py                      翻译训练循环
  inference.py                     翻译 checkpoint 加载和贪心解码
  classifier.py                    Transformer encoder 分类模型
  classification_data.py           AG News 数据读取
  classification_training.py       AG News 训练循环
  classification_inference.py      AG News 推理
  chinese_sentiment_data.py        ChnSentiCorp 数据读取
  chinese_sentiment_training.py    中文情感分类训练循环
  chinese_sentiment_inference.py   中文情感分类推理

scripts/
  download_multi30k.py             下载 Multi30k
  download_tatoeba_en_zh.py        下载 OPUS Tatoeba 英中语料
  train_translation.py             翻译训练
  evaluate_translation.py          翻译测试集评估
  translate.py                     翻译单句推理
  download_ag_news.py              下载 AG News
  train_classification.py          AG News 训练
  evaluate_classification.py       AG News 测试集评估
  classify.py                      AG News 单句推理
  download_chnsenticorp.py         下载 ChnSentiCorp
  train_chinese_sentiment.py       中文情感分类训练
  evaluate_chinese_sentiment.py    中文情感分类测试集评估
  classify_chinese_sentiment.py    中文情感分类单句推理
```

## 任务一：英德翻译

下载数据：

```bash
python scripts/download_multi30k.py --root dataset/multi30k
```

启动训练：

```bash
python scripts/train_translation.py \
  --data-root dataset/multi30k \
  --src-lang en \
  --tgt-lang de \
  --save-dir checkpoints/multi30k_en_de \
  --epochs 20 \
  --batch-size 128 \
  --max-len 128 \
  --d-model 256 \
  --heads 4 \
  --encoder-layers 3 \
  --decoder-layers 3 \
  --d-ff 1024 \
  --dropout 0.1 \
  --src-vocab-size 8000 \
  --tgt-vocab-size 8000 \
  --device auto
```

测试集评估：

```bash
python scripts/evaluate_translation.py \
  --checkpoint checkpoints/multi30k_en_de/best.pt \
  --data-root dataset/multi30k \
  --src-lang en \
  --tgt-lang de \
  --split test \
  --batch-size 128 \
  --device auto
```

单句推理：

```bash
python scripts/translate.py \
  --checkpoint checkpoints/multi30k_en_de/best.pt \
  --text "a man is riding a bike" \
  --device auto
```

## 任务二：英中翻译

下载数据：

```bash
python scripts/download_tatoeba_en_zh.py \
  --root dataset/tatoeba_en_zh \
  --max-pairs 30000 \
  --valid-size 1000 \
  --test-size 1000
```

启动训练：

```bash
python scripts/train_translation.py \
  --data-root dataset/tatoeba_en_zh \
  --src-lang en \
  --tgt-lang zh \
  --save-dir checkpoints/tatoeba_en_zh \
  --epochs 30 \
  --batch-size 128 \
  --max-len 80 \
  --d-model 256 \
  --heads 4 \
  --encoder-layers 3 \
  --decoder-layers 3 \
  --d-ff 1024 \
  --dropout 0.1 \
  --src-vocab-size 12000 \
  --tgt-vocab-size 8000 \
  --tgt-char-level \
  --device auto
```

测试集评估：

```bash
python scripts/evaluate_translation.py \
  --checkpoint checkpoints/tatoeba_en_zh/best.pt \
  --data-root dataset/tatoeba_en_zh \
  --src-lang en \
  --tgt-lang zh \
  --split test \
  --batch-size 128 \
  --device auto
```

单句推理：

```bash
python scripts/translate.py \
  --checkpoint checkpoints/tatoeba_en_zh/best.pt \
  --text "I like learning new languages." \
  --device auto
```

说明：当前中文端默认字符级 token，推理输出可能会在中文字符之间带空格。后续可以加中文 detokenize。

## 任务三：英文新闻分类

下载数据：

```bash
python scripts/download_ag_news.py --root dataset/ag_news
```

启动训练：

```bash
python scripts/train_classification.py \
  --data-root dataset/ag_news \
  --save-dir checkpoints/ag_news_classifier \
  --epochs 10 \
  --batch-size 128 \
  --max-len 128 \
  --d-model 256 \
  --heads 4 \
  --layers 3 \
  --d-ff 1024 \
  --dropout 0.1 \
  --vocab-size 30000 \
  --device auto
```

测试集评估：

```bash
python scripts/evaluate_classification.py \
  --checkpoint checkpoints/ag_news_classifier/best.pt \
  --data-root dataset/ag_news \
  --split test \
  --batch-size 128 \
  --device auto
```

单句推理：

```bash
python scripts/classify.py \
  --checkpoint checkpoints/ag_news_classifier/best.pt \
  --text "Apple shares rose after the company reported stronger quarterly revenue." \
  --device auto
```

## 任务四：中文情感分类

下载数据：

```bash
python scripts/download_chnsenticorp.py --root dataset/chnsenticorp
```

启动训练：

```bash
python scripts/train_chinese_sentiment.py \
  --data-root dataset/chnsenticorp \
  --save-dir checkpoints/chnsenticorp \
  --epochs 10 \
  --batch-size 128 \
  --max-len 256 \
  --d-model 256 \
  --heads 4 \
  --layers 3 \
  --d-ff 1024 \
  --dropout 0.1 \
  --vocab-size 8000 \
  --device auto
```

测试集评估：

```bash
python scripts/evaluate_chinese_sentiment.py \
  --checkpoint checkpoints/chnsenticorp/best.pt \
  --data-root dataset/chnsenticorp \
  --split test \
  --batch-size 128 \
  --device auto
```

单句推理：

```bash
python scripts/classify_chinese_sentiment.py \
  --checkpoint checkpoints/chnsenticorp/best.pt \
  --text "房间很干净，服务也很好，下次还会再来。" \
  --device auto
```

## 3090 参数建议

- 翻译任务比分类任务更吃显存，建议先用 `d-model=256`、`layers=3` 跑通。
- 如果显存不足，优先降低 `--batch-size`，再降低 `--max-len`。
- 英中翻译建议先使用 `--tgt-char-level`，避免额外中文分词依赖。
- 分类任务在 3090 上可以把 batch size 提高到 256，但先用 128 更稳。
- 如果验证集 loss 明显高于训练集 loss，可以尝试 `--dropout 0.2`。

## 数据目录

数据集和模型权重不会提交到 Git：

```text
dataset/
checkpoints/
```

需要在新机器上重新运行对应的下载脚本。
