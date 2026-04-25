# Transformer 部署包

这个目录包含模型权重、词表、配置和最小推理代码。

## 安装依赖

```bash
bash deploy.sh
```

如果不想创建虚拟环境：

```bash
SKIP_VENV=1 bash deploy.sh
```

## 推理示例

```bash
python predict.py --task en_de_translation --text "a man is riding a bike" --device cpu
python predict.py --task en_zh_translation --text "I like learning new languages." --device cpu
python predict.py --task ag_news --text "Apple shares rose after revenue beat expectations." --device cpu
python predict.py --task zh_sentiment --text "房间很干净，服务也很好。" --device cpu
```

## 已包含模型

- `multi30k_en_de`: 英德翻译
- `tatoeba_en_zh`: 英中翻译
- `ag_news_classifier`: 英文新闻分类
- `chnsenticorp`: 中文情感分类
