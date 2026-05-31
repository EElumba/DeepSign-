# python/sign_classifier.py
# WLASL TGCN sign classifier — FastAPI server on port 8001
#
# Setup (run once):
#   python -m venv .venv-reverse
#   .venv-reverse/Scripts/pip install -r requirements_reverse.txt
#
# Download pretrained weights (run once):
#   .venv-reverse/Scripts/python -c "
#   from huggingface_hub import snapshot_download
#   snapshot_download(
#       repo_id='sharonn18/tgcn-wlasl',
#       local_dir='models/tgcn_wlasl100',
#       allow_patterns=['asl100/*', 'configs/asl100.ini', 'tgcn_model.py', 'configs.py']
#   )
#   print('Downloaded TGCN weights')
#   "
#
# Run:
#   .venv-reverse/Scripts/uvicorn sign_classifier:app --port 8001

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models', 'tgcn_wlasl100'))

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="WLASL TGCN Classifier", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# WLASL100 gloss labels — top 100 most common ASL words
WLASL100_LABELS = [
    "book", "drink", "computer", "before", "chair", "go", "clothes", "who",
    "candy", "cousin", "deaf", "fine", "help", "no", "thin", "walk", "year",
    "yes", "all", "black", "cool", "finish", "hot", "like", "many", "mother",
    "now", "orange", "table", "thanksgiving", "what", "white", "wrong",
    "accident", "apple", "bird", "change", "color", "corn", "cow", "dance",
    "dark", "day", "doctor", "dog", "eat", "every", "family", "fast", "fish",
    "forget", "give", "glass", "good", "gray", "green", "happy", "hat",
    "hearing", "horse", "kiss", "language", "last", "letter", "man", "money",
    "month", "more", "name", "need", "nurse", "old", "pay", "pizza", "play",
    "purple", "right", "school", "secretary", "short", "shower", "son",
    "sorry", "spend", "spring", "store", "student", "tall", "tell", "thursday",
    "time", "uncle", "want", "water", "woman", "work", "world", "write",
]

model = None
model_load_error = None

def load_model():
    global model, model_load_error
    try:
        from tgcn_model import TGCN
        from configs import load_config

        cfg = load_config('models/tgcn_wlasl100/configs/asl100.ini')
        m = TGCN(cfg)
        weights = torch.load(
            'models/tgcn_wlasl100/asl100/pytorch_model.bin',
            map_location='cpu',
        )
        m.load_state_dict(weights)
        m.eval()
        model = m
        print("[sign_classifier] TGCN model loaded — WLASL100 ready")
    except Exception as e:
        model_load_error = str(e)
        print(f"[sign_classifier] WARNING: model not loaded — {e}")
        print("[sign_classifier] Run the download script above and restart.")

load_model()


class ClassifyRequest(BaseModel):
    frames: List[List[float]]   # shape (N, 165) — N up to 30


@app.post("/classify")
async def classify(req: ClassifyRequest):
    if model is None:
        return {"gloss": None, "confidence": 0.0, "error": f"Model not loaded: {model_load_error}"}

    if len(req.frames) < 10:
        return {"gloss": None, "confidence": 0.0, "error": "not enough frames"}

    # Trim or pad to exactly 30 frames
    frames = req.frames[-30:] if len(req.frames) > 30 else req.frames
    while len(frames) < 30:
        frames.append(frames[-1])

    x = torch.tensor([frames], dtype=torch.float32)   # (1, 30, 165)

    with torch.no_grad():
        logits = model(x)                              # (1, num_classes)
        probs  = torch.softmax(logits, dim=-1)
        confidence, pred_idx = probs.max(dim=-1)

    gloss = WLASL100_LABELS[pred_idx.item()]
    conf  = round(confidence.item(), 4)

    top3 = [
        {"gloss": WLASL100_LABELS[i], "confidence": round(probs[0][i].item(), 3)}
        for i in probs[0].topk(3).indices.tolist()
    ]

    return {"gloss": gloss, "confidence": conf, "top3": top3}


@app.get("/health")
async def health():
    return {
        "status": "ok" if model is not None else "model_missing",
        "vocab_size": len(WLASL100_LABELS),
        "model_loaded": model is not None,
        "error": model_load_error,
    }
