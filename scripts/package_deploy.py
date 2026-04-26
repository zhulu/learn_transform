from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


TASKS = {
    "multi30k_en_de": {
        "kind": "translation",
        "required": ["best.pt", "config.json", "src_vocab.json", "tgt_vocab.json"],
        "description": "英德翻译",
    },
    "tatoeba_en_zh": {
        "kind": "translation",
        "required": ["best.pt", "config.json", "src_vocab.json", "tgt_vocab.json"],
        "description": "英中翻译",
    },
    "lcsts_summary": {
        "kind": "translation",
        "required": ["best.pt", "config.json", "src_vocab.json", "tgt_vocab.json"],
        "description": "中文中心思想/摘要生成",
    },
    "ag_news_classifier": {
        "kind": "classification",
        "required": ["best.pt", "config.json", "vocab.json"],
        "description": "英文新闻分类",
    },
    "chnsenticorp": {
        "kind": "classification",
        "required": ["best.pt", "config.json", "vocab.json"],
        "description": "中文情感分类",
    },
}


def copy_tree(src: Path, dst: Path) -> None:
    """复制目录，跳过 __pycache__。"""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def copy_task(checkpoint_root: Path, output_models: Path, task_name: str, allow_missing: bool) -> dict:
    """复制单个任务部署所需的 checkpoint、config 和词表。"""
    task = TASKS[task_name]
    src_dir = checkpoint_root / task_name
    missing = [name for name in task["required"] if not (src_dir / name).exists()]
    if missing:
        message = f"{task_name} missing files: {missing}"
        if allow_missing:
            print(f"skip: {message}")
            return {"name": task_name, "skipped": True, "missing": missing}
        raise FileNotFoundError(message)

    dst_dir = output_models / task_name
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in task["required"]:
        shutil.copy2(src_dir / name, dst_dir / name)

    return {
        "name": task_name,
        "kind": task["kind"],
        "description": task["description"],
        "files": task["required"],
        "skipped": False,
    }


def write_deploy_readme(output: Path, manifest: dict) -> None:
    """写部署包说明。"""
    content = [
        "# Transformer 部署包",
        "",
        "这个目录包含模型权重、词表、配置和最小推理代码。",
        "",
        "## 安装依赖",
        "",
        "```bash",
        "bash deploy.sh",
        "```",
        "",
        "如果不想创建虚拟环境：",
        "",
        "```bash",
        "SKIP_VENV=1 bash deploy.sh",
        "```",
        "",
        "## 推理示例",
        "",
        "```bash",
        "python predict.py --task en_de_translation --text \"a man is riding a bike\" --device cpu",
        "python predict.py --task en_zh_translation --text \"I like learning new languages.\" --device cpu",
        "python predict.py --task ag_news --text \"Apple shares rose after revenue beat expectations.\" --device cpu",
        "python predict.py --task zh_sentiment --text \"房间很干净，服务也很好。\" --device cpu",
        "python predict.py --task lcsts_summary --text \"国务院新闻办公室今天举行发布会，介绍当前经济运行情况。\" --device cpu --max-len 40",
        "```",
        "",
        "## 已包含模型",
        "",
    ]
    for item in manifest["tasks"]:
        if item.get("skipped"):
            content.append(f"- `{item['name']}`: skipped, missing={item['missing']}")
        else:
            content.append(f"- `{item['name']}`: {item['description']}")
    content.append("")
    (output / "README_DEPLOY.md").write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Package trained checkpoints and inference code for deployment.")
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument("--output", default="deploy_package")
    parser.add_argument("--allow-missing", action="store_true", help="只打包已经训练完成的任务")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    checkpoint_root = Path(args.checkpoint_root)
    output = Path(args.output)

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    copy_tree(root / "src", output / "src")
    shutil.copy2(root / "requirements.txt", output / "requirements.txt")
    shutil.copy2(root / "scripts" / "deploy_predict.py", output / "predict.py")
    shutil.copy2(root / "scripts" / "deploy.sh", output / "deploy.sh")

    output_models = output / "models"
    manifest = {"tasks": []}
    for task_name in TASKS:
        manifest["tasks"].append(copy_task(checkpoint_root, output_models, task_name, args.allow_missing))

    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_deploy_readme(output, manifest)
    print(f"deploy package ready: {output}")


if __name__ == "__main__":
    main()
