# learn_transform

这是一个从零开始学习 Transformer 的小工程，尽量只依赖 PyTorch，把核心结构和训练流程写清楚。

目前包含两个任务：

1. **机器翻译**：使用 encoder-decoder Transformer，在 Multi30k 上做英文到德文翻译。
2. **文本分类**：使用 encoder-only Transformer，在 AG News 上做新闻主题四分类。

项目目标不是追求最高指标，而是把 Transformer 的数据流、mask、训练循环、checkpoint、推理流程完整跑通。

## 项目结构

```text
src/
  vocab.py                    词表构建、token 和 id 的互相转换
  data.py                     翻译任务的数据读取、分词、padding batch
  model.py                    手写 Transformer encoder-decoder
  training.py                 翻译任务训练循环
  inference.py                翻译任务 checkpoint 加载和贪心解码
  classification_data.py      文本分类数据读取、词表构建、padding batch
  classifier.py               Transformer encoder 文本分类模型
  classification_training.py  文本分类训练循环
  classification_inference.py 文本分类 checkpoint 加载和单句预测

scripts/
  download_multi30k.py        下载 Multi30k 翻译数据集
  train_translation.py        翻译任务训练入口
  translate.py                翻译任务推理入口
  download_ag_news.py         下载 AG News 文本分类数据集
  train_classification.py     文本分类训练入口
  classify.py                 文本分类推理入口
```

## 环境准备

```powershell
pip install -r requirements.txt
```

如果你要用 3090 训练，需要安装 CUDA 版 PyTorch。可以先检查：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

看到 `True` 和显卡名称后，训练脚本的 `--device auto` 就会自动使用 GPU。

## 任务一：机器翻译

### 下载 Multi30k

```powershell
python scripts\download_multi30k.py --root dataset\multi30k
```

下载后会得到：

```text
dataset/multi30k/train.en
dataset/multi30k/train.de
dataset/multi30k/valid.en
dataset/multi30k/valid.de
dataset/multi30k/test.en
dataset/multi30k/test.de
```

### 3090 推荐训练参数

第一版建议用小模型，方便确认 loss 能稳定下降：

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

如果显存充足，可以稍微放大：

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

### 翻译推理

```powershell
python scripts\translate.py `
  --checkpoint checkpoints\multi30k_en_de\best.pt `
  --text "a man is riding a bike"
```

## 任务二：文本分类

文本分类使用 AG News 数据集，共四类：

```text
World
Sports
Business
Sci/Tech
```

这个任务只使用 Transformer encoder，不使用 decoder。它能帮助你理解另一种常见用法：先用 encoder 得到整句表示，再接一个分类头。

### 下载 AG News

```powershell
python scripts\download_ag_news.py --root dataset\ag_news
```

下载后会得到：

```text
dataset/ag_news/train.csv
dataset/ag_news/test.csv
```

### 3090 推荐训练参数

```powershell
python scripts\train_classification.py `
  --data-root dataset\ag_news `
  --save-dir checkpoints\ag_news_classifier `
  --epochs 10 `
  --batch-size 128 `
  --max-len 128 `
  --d-model 256 `
  --heads 4 `
  --layers 3 `
  --d-ff 1024
```

如果想先快速试跑，可以降低模型：

```powershell
python scripts\train_classification.py `
  --data-root dataset\ag_news `
  --save-dir checkpoints\ag_news_classifier_small `
  --epochs 3 `
  --batch-size 256 `
  --max-len 96 `
  --d-model 128 `
  --heads 4 `
  --layers 2 `
  --d-ff 512
```

### 分类推理

```powershell
python scripts\classify.py `
  --checkpoint checkpoints\ag_news_classifier\best.pt `
  --text "Apple shares rose after the company reported stronger quarterly revenue."
```

输出会包含预测类别、最高类别概率，以及四个类别各自的概率。

## 两个任务的区别

翻译任务：

- 使用完整 encoder-decoder Transformer。
- encoder 读源语言句子。
- decoder 自回归生成目标语言句子。
- 训练时使用 teacher forcing。
- 推理时使用贪心解码。

文本分类任务：

- 只使用 Transformer encoder。
- 输入整段新闻文本。
- 对非 padding token 的 hidden state 做平均池化。
- 分类头输出四个类别的 logits。

## 后续扩展方向

- 把翻译数据从 Multi30k 英德替换成 IWSLT 英中。
- 给翻译任务加 BLEU 评估。
- 给翻译推理加 beam search。
- 文本分类任务改成中文新闻分类或情感分类。
- 把当前简单词表升级为 BPE/SentencePiece。
