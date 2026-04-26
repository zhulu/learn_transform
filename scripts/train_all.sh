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

MULTI30K_EPOCHS="${MULTI30K_EPOCHS:-30}"
TATOEBA_EPOCHS="${TATOEBA_EPOCHS:-50}"
AG_NEWS_EPOCHS="${AG_NEWS_EPOCHS:-15}"
CHNSENTI_EPOCHS="${CHNSENTI_EPOCHS:-15}"
LCSTS_EPOCHS="${LCSTS_EPOCHS:-25}"
LCSTS_TRAIN_SIZE="${LCSTS_TRAIN_SIZE:-200000}"
LCSTS_VALID_SIZE="${LCSTS_VALID_SIZE:-4000}"
LCSTS_TEST_SIZE="${LCSTS_TEST_SIZE:-4000}"
LCSTS_BATCH="${LCSTS_BATCH:-64}"

has_files() {
  local file
  for file in "$@"; do
    if [[ ! -s "${file}" ]]; then
      return 1
    fi
  done
  return 0
}

echo "==> 下载/准备数据"
if has_files \
  "${DATA_ROOT}/multi30k/train.en" \
  "${DATA_ROOT}/multi30k/train.de" \
  "${DATA_ROOT}/multi30k/valid.en" \
  "${DATA_ROOT}/multi30k/valid.de" \
  "${DATA_ROOT}/multi30k/test.en" \
  "${DATA_ROOT}/multi30k/test.de"; then
  echo "==> Multi30k 已存在，跳过下载：${DATA_ROOT}/multi30k"
else
  python scripts/download_multi30k.py --root "${DATA_ROOT}/multi30k"
fi

if has_files \
  "${DATA_ROOT}/tatoeba_en_zh/train.en" \
  "${DATA_ROOT}/tatoeba_en_zh/train.zh" \
  "${DATA_ROOT}/tatoeba_en_zh/valid.en" \
  "${DATA_ROOT}/tatoeba_en_zh/valid.zh" \
  "${DATA_ROOT}/tatoeba_en_zh/test.en" \
  "${DATA_ROOT}/tatoeba_en_zh/test.zh"; then
  echo "==> Tatoeba 英中语料已存在，跳过下载：${DATA_ROOT}/tatoeba_en_zh"
else
  python scripts/download_tatoeba_en_zh.py \
    --root "${DATA_ROOT}/tatoeba_en_zh" \
    --max-pairs 30000 \
    --valid-size 1000 \
    --test-size 1000
fi

if has_files \
  "${DATA_ROOT}/ag_news/train.csv" \
  "${DATA_ROOT}/ag_news/test.csv"; then
  echo "==> AG News 已存在，跳过下载：${DATA_ROOT}/ag_news"
else
  python scripts/download_ag_news.py --root "${DATA_ROOT}/ag_news"
fi

if has_files \
  "${DATA_ROOT}/chnsenticorp/train.tsv" \
  "${DATA_ROOT}/chnsenticorp/valid.tsv" \
  "${DATA_ROOT}/chnsenticorp/test.tsv"; then
  echo "==> ChnSentiCorp 已存在，跳过下载：${DATA_ROOT}/chnsenticorp"
else
  python scripts/download_chnsenticorp.py --root "${DATA_ROOT}/chnsenticorp"
fi

if has_files \
  "${DATA_ROOT}/lcsts_summary/train.src" \
  "${DATA_ROOT}/lcsts_summary/train.tgt" \
  "${DATA_ROOT}/lcsts_summary/valid.src" \
  "${DATA_ROOT}/lcsts_summary/valid.tgt" \
  "${DATA_ROOT}/lcsts_summary/test.src" \
  "${DATA_ROOT}/lcsts_summary/test.tgt"; then
  echo "==> LCSTS 摘要语料已存在，跳过下载：${DATA_ROOT}/lcsts_summary"
else
  python scripts/download_lcsts_summary.py \
    --root "${DATA_ROOT}/lcsts_summary" \
    --train-size "${LCSTS_TRAIN_SIZE}" \
    --valid-size "${LCSTS_VALID_SIZE}" \
    --test-size "${LCSTS_TEST_SIZE}"
fi

echo "==> 训练任务一：英德翻译"
python scripts/train_translation.py \
  --data-root "${DATA_ROOT}/multi30k" \
  --src-lang en \
  --tgt-lang de \
  --save-dir "${CKPT_ROOT}/multi30k_en_de" \
  --epochs "${MULTI30K_EPOCHS}" \
  --batch-size "${TRANSLATION_BATCH}" \
  --max-len 128 \
  --d-model 512 \
  --heads 8 \
  --encoder-layers 6 \
  --decoder-layers 6 \
  --d-ff 2048 \
  --dropout 0.1 \
  --src-vocab-size 10000 \
  --tgt-vocab-size 10000 \
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
  --d-model 512 \
  --heads 8 \
  --encoder-layers 6 \
  --decoder-layers 6 \
  --d-ff 2048 \
  --dropout 0.1 \
  --src-vocab-size 16000 \
  --tgt-vocab-size 10000 \
  --tgt-char-level \
  --device "${DEVICE}"

echo "==> 训练任务三：英文新闻分类"
python scripts/train_classification.py \
  --data-root "${DATA_ROOT}/ag_news" \
  --save-dir "${CKPT_ROOT}/ag_news_classifier" \
  --epochs "${AG_NEWS_EPOCHS}" \
  --batch-size "${CLASSIFICATION_BATCH}" \
  --max-len 128 \
  --d-model 512 \
  --heads 8 \
  --layers 6 \
  --d-ff 2048 \
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
  --d-model 512 \
  --heads 8 \
  --layers 6 \
  --d-ff 2048 \
  --dropout 0.1 \
  --vocab-size 10000 \
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
  --d-model 512 \
  --heads 8 \
  --encoder-layers 6 \
  --decoder-layers 6 \
  --d-ff 2048 \
  --dropout 0.1 \
  --src-vocab-size 16000 \
  --tgt-vocab-size 10000 \
  --src-char-level \
  --tgt-char-level \
  --device "${DEVICE}"

echo "==> 五个任务训练完成"
echo "模型目录：${CKPT_ROOT}"
echo "下一步可运行：python scripts/package_deploy.py --checkpoint-root ${CKPT_ROOT} --output deploy_package"
