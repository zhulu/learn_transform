# learn_transform

这是一个从零开始学习 Transformer 的小工程。代码尽量只依赖 PyTorch，把词表、数据读取、mask、模型结构、训练循环、测试评估和推理流程都显式写出来。

当前包含五个任务，默认参数都按 RTX/ATX 3090 24G 显存做了保守设计：

1. **英德翻译**：Multi30k，encoder-decoder Transformer。
2. **英中翻译**：OPUS Tatoeba Mandarin Chinese-English 小语料，中文端默认字符级 token。
3. **英文新闻分类**：AG News，encoder-only Transformer。
4. **中文情感分类**：ChnSentiCorp，中文评论二分类，默认字符级 token。
5. **中文中心思想/摘要生成**：LCSTS，复用 encoder-decoder seq2seq 训练接口。

## 环境准备

```bash
pip install -r requirements.txt
```

## 项目结构

```text
src/              # 核心源码 (模型定义、训练推理逻辑)
scripts/          # 任务脚本 (数据下载、单任务训练、评估)
web/              # Web 演示服务 (基于 FastAPI)
deploy_package/   # 打包好的部署包 (包含模型权重和推理源码)
```

## Web 演示服务 (Transformer Lab)

为了更直观地展示模型效果，项目包含了一个交互式 Web 演示界面。

### 功能亮点
- **模型规格卡片**：实时展示当前任务的模型参数（Params）、隐藏层维度、层数及训练语料规模。
- **可视化推理**：
    - 翻译任务：实时生成译文。
    - 摘要任务：输入中文短文，生成中心思想。
    - 分类任务：以进度条形式展示 Softmax 置信度分数。
- **轻量化部署**：针对 CPU 环境优化，支持模型按需加载。

### 启动方式
1. 安装依赖：`pip install -r web/requirements.txt`
2. 启动服务：`python web/main.py`
3. 访问地址：`http://localhost:8000`

## 核心任务说明

(各任务具体训练和评估命令请参考 scripts/ 目录下的脚本或原 README 详细文档)

## 一键训练

顺序下载数据并训练五个任务：

```bash
bash scripts/train_all.sh
```

任务 5 的常用覆盖参数：

```bash
LCSTS_TRAIN_SIZE=100000 \
LCSTS_VALID_SIZE=4000 \
LCSTS_TEST_SIZE=4000 \
LCSTS_EPOCHS=20 \
LCSTS_BATCH=64 \
bash scripts/train_all.sh
```

## 可选任务：中文中心思想/摘要生成

这个任务使用 LCSTS 中文短文本摘要数据，字段含义为：

```text
text     原文
summary  中心思想/摘要/标题式概括
```

下载并整理语料：

```bash
python scripts/download_lcsts_summary.py \
  --root dataset/lcsts_summary \
  --train-size 100000 \
  --valid-size 4000 \
  --test-size 4000
```

如果遇到 Hugging Face `429 Too Many Requests` 限流，脚本会自动重试并默认断点续传。再次运行同一命令即可继续下载。想清空重下可以加：

```bash
python scripts/download_lcsts_summary.py \
  --root dataset/lcsts_summary \
  --train-size 100000 \
  --valid-size 4000 \
  --test-size 4000 \
  --overwrite
```

脚本会生成现有 seq2seq 接口可直接使用的文件：

```text
dataset/lcsts_summary/train.src
dataset/lcsts_summary/train.tgt
dataset/lcsts_summary/valid.src
dataset/lcsts_summary/valid.tgt
dataset/lcsts_summary/test.src
dataset/lcsts_summary/test.tgt
```

3090 推荐训练命令：

```bash
python scripts/train_translation.py \
  --data-root dataset/lcsts_summary \
  --src-lang src \
  --tgt-lang tgt \
  --save-dir checkpoints/lcsts_summary \
  --epochs 20 \
  --batch-size 64 \
  --max-len 256 \
  --d-model 256 \
  --heads 4 \
  --encoder-layers 3 \
  --decoder-layers 3 \
  --d-ff 1024 \
  --dropout 0.2 \
  --src-vocab-size 12000 \
  --tgt-vocab-size 8000 \
  --src-char-level \
  --tgt-char-level \
  --device auto
```

测试集评估：

```bash
python scripts/evaluate_translation.py \
  --checkpoint checkpoints/lcsts_summary/best.pt \
  --data-root dataset/lcsts_summary \
  --src-lang src \
  --tgt-lang tgt \
  --split test \
  --batch-size 64 \
  --device auto
```

单条文章生成中心思想：

```bash
python scripts/translate.py \
  --checkpoint checkpoints/lcsts_summary/best.pt \
  --text "国务院新闻办公室今天举行发布会，介绍当前经济运行情况和下一阶段政策安排。" \
  --max-len 40 \
  --device auto
```

接口结论：暂时不需要修改训练、测试接口。中心思想生成和翻译一样都是 `输入序列 -> 输出序列`，可以直接复用 `train_translation.py`、`evaluate_translation.py` 和 `translate.py`。后续如果要更严谨评估摘要质量，再单独加 ROUGE/chrF 等摘要指标。

## 打包部署

使用 `scripts/package_deploy.py` 可以将训练好的 `best.pt` 权重与必要的推理源码打包到一个独立目录中，方便在无源码环境或生产环境中部署。

```bash
python scripts/package_deploy.py --checkpoint-root checkpoints --output deploy_package
```

部署包统一推理入口已支持任务 5：

```bash
python predict.py \
  --task lcsts_summary \
  --text "国务院新闻办公室今天举行发布会，介绍当前经济运行情况和下一阶段政策安排。" \
  --device cpu \
  --max-len 40
```

## 项目优化记录

### 2026-04-26 优化更新 (By Gemini CLI)

为了显著提升模型在五个核心任务上的实际应用效果，对项目进行了如下整体优化：

1.  **架构升级：Pre-LN Transformer**
    *   将模型结构由 Post-LN 重构为 **Pre-LN**（前置层归一化）。
    *   Pre-LN 架构在训练深层 Transformer 时具有更强的稳定性，能有效缓解梯度消失/爆炸问题，是目前大模型的主流选择。

2.  **训练策略优化：OneCycleLR 调度器**
    *   引入了 `OneCycleLR` 学习率调度机制，支持 Warmup（预热）和余弦退火。
    *   动态调整训练过程中的学习率，帮助模型更快跳出局部最优，并实现更平滑的最终收敛。

3.  **模型规模全面提升**
    *   **参数量级**：将核心隐藏层维度 `d_model` 从 256 提升至 **512**，多头注意力头数 `heads` 从 4 提升至 **8**。
    *   **深度增加**：Transformer 层数从 3 层翻倍至 **6 层**。
    *   **前馈网络**：`d_ff` 从 1024 提升至 **2048**。

4.  **训练强度增强**
    *   增加了所有任务的训练轮次（Epochs）。
    *   扩大了部分任务的词表容量和训练数据规模（如 LCSTS 摘要任务训练集提升至 20 万）。

5.  **兼容性修复**
    *   优化了 `run_epoch` 接口，在支持训练调度的同时，保持了与现有测试集评估脚本的向后兼容。

> **注**：由于架构改动，旧版本的 `best.pt` 权重已不再适用，建议重新运行 `bash scripts/train_all.sh` 以获取最新性能的模型。

## 数据与模型

数据集和模型权重默认不提交：
- `dataset/` (原始数据)
- `checkpoints/` (训练产物)
- `*.tar`, `*.gz` (压缩包)
