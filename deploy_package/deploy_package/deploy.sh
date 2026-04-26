#!/usr/bin/env bash
set -euo pipefail

# 在第三方设备的部署包根目录中运行：
# bash deploy.sh
#
# 默认创建 .venv 并安装 requirements.txt。若设备已有环境，可设置 SKIP_VENV=1。

PYTHON_BIN="${PYTHON_BIN:-python3}"
SKIP_VENV="${SKIP_VENV:-0}"

if [[ "${SKIP_VENV}" != "1" ]]; then
  "${PYTHON_BIN}" -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "部署环境准备完成。可用命令示例："
echo "python predict.py --task en_de_translation --text \"a man is riding a bike\" --device cpu"
echo "python predict.py --task en_zh_translation --text \"I like learning new languages.\" --device cpu"
echo "python predict.py --task ag_news --text \"Apple shares rose after revenue beat expectations.\" --device cpu"
echo "python predict.py --task zh_sentiment --text \"房间很干净，服务也很好。\" --device cpu"
echo "python predict.py --task lcsts_summary --text \"国务院新闻办公室今天举行发布会，介绍当前经济运行情况和下一阶段政策安排。\" --device cpu --max-len 40"
