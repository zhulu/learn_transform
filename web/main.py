import sys
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Dict, Any

# Add deploy_package to sys.path
# Looking at the structure: D:\project\learn_transform\deploy_package\deploy_package\src
PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "deploy_package" / "deploy_package"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

try:
    from src.chinese_sentiment_inference import (
        load_chinese_sentiment_checkpoint,
        predict_chinese_sentiment,
    )
    from src.classification_inference import classify_text, load_classifier_checkpoint
    from src.inference import greedy_translate, load_checkpoint
except ImportError as e:
    print(f"Error importing from deploy_package: {e}")
    print(f"PACKAGE_ROOT: {PACKAGE_ROOT}")
    print(f"Contents of PACKAGE_ROOT: {list(PACKAGE_ROOT.glob('*')) if PACKAGE_ROOT.exists() else 'Does not exist'}")

app = FastAPI(title="Transformer Web Demo")

# Cache for loaded models
model_cache = {}

TASK_MODEL_DIRS = {
    "en_de_translation": "multi30k_en_de",
    "en_zh_translation": "tatoeba_en_zh",
    "ag_news": "ag_news_classifier",
    "zh_sentiment": "chnsenticorp",
}

class PredictRequest(BaseModel):
    task: str
    text: str

def get_model(task: str):
    if task in model_cache:
        return model_cache[task]
    
    checkpoint_path = PACKAGE_ROOT / "models" / TASK_MODEL_DIRS[task] / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    device = "cpu"
    
    if task in {"en_de_translation", "en_zh_translation"}:
        model_data = load_checkpoint(checkpoint_path, device)
        model_cache[task] = model_data
    elif task == "ag_news":
        model_data = load_classifier_checkpoint(checkpoint_path, device)
        model_cache[task] = model_data
    elif task == "zh_sentiment":
        model_data = load_chinese_sentiment_checkpoint(checkpoint_path, device)
        model_cache[task] = model_data
        
    return model_cache[task]

@app.post("/predict")
async def predict(request: PredictRequest):
    if request.task not in TASK_MODEL_DIRS:
        raise HTTPException(status_code=400, detail="Invalid task")
    
    try:
        if request.task in {"en_de_translation", "en_zh_translation"}:
            model, src_vocab, tgt_vocab, config, device = get_model(request.task)
            output = greedy_translate(
                model,
                src_vocab,
                tgt_vocab,
                request.text,
                device,
                max_len=128,
                lowercase=config.lowercase,
                src_char_level=config.src_char_level,
            )
            return {"result": output}
        
        elif request.task == "ag_news":
            model, vocab, labels, config, device = get_model(request.task)
            label, prob, scores = classify_text(
                model,
                vocab,
                labels,
                request.text,
                device,
                max_len=config.max_len,
                lowercase=config.lowercase,
            )
            return {
                "label": label,
                "probability": float(prob),
                "scores": {name: float(score) for name, score in scores}
            }
        
        elif request.task == "zh_sentiment":
            model, vocab, labels, config, device = get_model(request.task)
            label, prob, scores = predict_chinese_sentiment(
                model,
                vocab,
                labels,
                request.text,
                device,
                max_len=config.max_len,
                lowercase=config.lowercase,
                char_level=config.char_level,
            )
            return {
                "label": label,
                "probability": float(prob),
                "scores": {name: float(score) for name, score in scores}
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_index():
    return FileResponse(Path(__file__).resolve().parent / "static" / "index.html")

# Serve static files
app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parent / "static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
