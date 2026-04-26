#!/usr/bin/env bash
set -euo pipefail

# 顺序训练五个任务：
# 1. Multi30k 英德翻译
# 2. Tatoeba 英中翻译
# 3. AG News 英文新闻分类
# 4. ChnSentiCorp 中文情感分类
# 5. LCSTS 中文中心思想/摘要生成
#
# 默认参数按 RTX/ATX 3090 24G 做保守设置。需要临时调整时，可以用环境变量覆盖：
# DEVICE=cuda TRANSLATION_BATCH=96 bash scripts/train_all.sh

DATA_ROOT="${DATA_ROOT:-dataset}"
CKPT_ROOT="${CKPT_ROOT:-checkpoints}"
DEVICE="${DEVICE:-auto}"

TRANSLATION_BATCH="${TRANSLATION_BATCH:-128}"
CLASSIFICATION_BATCH="${CLASSIFICATION_BATCH:-128}"

MULTI30K_EPOCHS="${MULTI30K_EPOCHS:-20}"
TATOEBA_EPOCHS="${TATOEBA_EPOCHS:-30}"
AG_NEWS_EPOCHS="${AG_NEWS_EPOCHS:-10}"
CHNSENTI_EPOCHS="${CHNSENTI_EPOCHS:-10}"
LCSTS_EPOCHS="${LCSTS_EPOCHS:-20}"
LCSTS_TRAIN_SIZE="${LCSTS_TRAIN_SIZE:-100000}"
LCSTS_VALID_SIZE="${LCSTS_VALID_SIZE:-4000}"
LCSTS_TEST_SIZE="${LCSTS_TEST_SIZE:-4000}"
LCSTS_BATCH="${LCSTS_BATCH:-64}"

echo "==> 下载/准备数据"
python scripts/download_multi30k.py --root "${DATA_ROOT}/multi30k"
python scripts/download_tatoeba_en_zh.py \
  --root "${DATA_ROOT}/tatoeba_en_zh" \
  --max-pairs 30000 \
  --valid-size 1000 \
  --test-size 1000
python scripts/download_ag_news.py --root "${DATA_ROOT}/ag_news"
python scripts/download_chnsenticorp.py --root "${DATA_ROOT}/chnsenticorp"
python scripts/download_lcsts_summary.py \
  --root "${DATA_ROOT}/lcsts_summary" \
  --train-size "${LCSTS_TRAIN_SIZE}" \
  --valid-size "${LCSTS_VALID_SIZE}" \
  --test-size "${LCSTS_TEST_SIZE}"

echo "==> 训练任务一：英德翻译"
python scripts/train_translation.py \
  --data-root "${DATA_ROOT}/multi30k" \
  --src-lang en \
  --tgt-lang de \
  --save-dir "${CKPT_ROOT}/multi30k_en_de" \
  --epochs "${MULTI30K_EPOCHS}" \
  --batch-size "${TRANSLATION_BATCH}" \
  --max-len 128 \
  --d-model 256 \
  --heads 4 \
  --encoder-layers 3 \
  --decoder-layers 3 \
  --d-ff 1024 \
  --dropout 0.1 \
  --src-vocab-size 8000 \
  --tgt-vocab-size 8000 \
  --device "${DEVICE}"

echo "==> 训练任务二：英中翻译"
python scripts/train_translation.py \
  --data-root "${DATA_ROOT}/tatoeba_en_zh" \
  --src-lang en \
  --tgt-lang zh \
  --save-dir "${CKPT_ROOT}/tatoeba_en_zh" \
  --epochs "${TATOEBA_EPOCHS}" \
  --batch-size "${TRANSLATION_BATCH}" \
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
  --device "${DEVICE}"

echo "==> 训练任务三：英文新闻分类"
python scripts/train_classification.py \
  --data-root "${DATA_ROOT}/ag_news" \
  --save-dir "${CKPT_ROOT}/ag_news_classifier" \
  --epochs "${AG_NEWS_EPOCHS}" \
  --batch-size "${CLASSIFICATION_BATCH}" \
  --max-len 128 \
  --d-model 256 \
  --heads 4 \
  --layers 3 \
  --d-ff 1024 \
  --dropout 0.1 \
  --vocab-size 30000 \
  --device "${DEVICE}"

echo "==> 训练任务四：中文情感分类"
python scripts/train_chinese_sentiment.py \
  --data-root "${DATA_ROOT}/chnsenticorp" \
  --save-dir "${CKPT_ROOT}/chnsenticorp" \
  --epochs "${CHNSENTI_EPOCHS}" \
  --batch-size "${CLASSIFICATION_BATCH}" \
  --max-len 256 \
  --d-model 256 \
  --heads 4 \
  --layers 3 \
  --d-ff 1024 \
  --dropout 0.1 \
  --vocab-size 8000 \
  --device "${DEVICE}"

echo "==> 训练任务五：中文中心思想/摘要生成"
python scripts/train_translation.py \
  --data-root "${DATA_ROOT}/lcsts_summary" \
  --src-lang src \
  --tgt-lang tgt \
  --save-dir "${CKPT_ROOT}/lcsts_summary" \
  --epochs "${LCSTS_EPOCHS}" \
  --batch-size "${LCSTS_BATCH}" \
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
  --device "${DEVICE}"

echo "==> 五个任务训练完成"
echo "模型目录：${CKPT_ROOT}"
echo "下一步可运行：python scripts/package_deploy.py --checkpoint-root ${CKPT_ROOT} --output deploy_package"
