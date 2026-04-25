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
    - 分类任务：以进度条形式展示 Softmax 置信度分数。
- **轻量化部署**：针对 CPU 环境优化，支持模型按需加载。

### 启动方式
1. 安装依赖：`pip install -r web/requirements.txt`
2. 启动服务：`python web/main.py`
3. 访问地址：`http://localhost:8000`

## 核心任务说明

(各任务具体训练和评估命令请参考 scripts/ 目录下的脚本或原 README 详细文档)

## 打包部署

使用 `scripts/package_deploy.py` 可以将训练好的 `best.pt` 权重与必要的推理源码打包到一个独立目录中，方便在无源码环境或生产环境中部署。

```bash
python scripts/package_deploy.py --checkpoint-root checkpoints --output deploy_package
```

## 数据与模型

数据集和模型权重默认不提交：
- `dataset/` (原始数据)
- `checkpoints/` (训练产物)
- `*.tar`, `*.gz` (压缩包)
