"""Font Classifier API.

Pipeline: image -> EasyOCR detection (upscale x2 + HSV value) -> word crops
          -> ResNet18 (HSV value + resize-with-padding) -> confidence-weighted vote -> font

Endpoints:
  GET  /health         service status
  POST /predict        one image    -> one prediction
  POST /predict_batch  several images -> list of predictions

Run:  MODEL_PATH=models/resnet18_final.pt uvicorn api:app --reload
Docs: http://127.0.0.1:8000/docs
"""
import io
import os
from contextlib import asynccontextmanager
from typing import List, Optional

import easyocr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from torchvision import models, transforms

from src.preprocessing import resize_pad, to_hsv_value, upscale

MODEL_PATH = os.getenv("MODEL_PATH", "models/resnet18_final.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

crop_tf = transforms.Compose([
    transforms.Lambda(to_hsv_value),
    transforms.Lambda(resize_pad),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and the OCR reader once, at startup."""
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(ckpt["classes"]))
    model.load_state_dict(ckpt["model"])
    state["model"] = model.to(DEVICE).eval()
    state["classes"] = ckpt["classes"]
    state["reader"] = easyocr.Reader(["en"], gpu=(DEVICE == "cuda"))
    yield
    state.clear()


app = FastAPI(title="Font Classifier API", lifespan=lifespan)


class Prediction(BaseModel):
    filename: str
    font: Optional[str]  # None when no text is detected
    confidence: float
    message: Optional[str] = None


class BatchResponse(BaseModel):
    predictions: List[Prediction]
    fonts: List[Optional[str]]


def predict_font(pil_img: Image.Image):
    """Return (font, confidence), or (None, 0.0) when no text is detected."""
    model, classes, reader = state["model"], state["classes"], state["reader"]
    det = to_hsv_value(upscale(pil_img, 2))
    results = reader.readtext(np.array(det), text_threshold=0.3, low_text=0.3)

    probs = torch.zeros(len(classes), device=DEVICE)
    n = 0
    for bbox, text, conf in results:
        if len(text.strip()) < 2:
            continue
        xs = [p[0] / 2 for p in bbox]  # back to original image coordinates
        ys = [p[1] / 2 for p in bbox]
        crop = pil_img.crop((max(int(min(xs)), 0), max(int(min(ys)), 0),
                             int(max(xs)), int(max(ys))))
        if crop.width < 3 or crop.height < 3:
            continue
        x = crop_tf(crop).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs += conf * F.softmax(model(x)[0], dim=0)  # weighted by OCR confidence
        n += 1

    if n == 0:
        return None, 0.0
    probs /= probs.sum()
    idx = int(probs.argmax())
    return classes[idx], float(probs[idx])


def _read_image(file: UploadFile, data: bytes) -> Image.Image:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, f"{file.filename}: file must be an image.")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"{file.filename}: unreadable image.") from exc


def _predict_one(file: UploadFile, data: bytes) -> Prediction:
    font, conf = predict_font(_read_image(file, data))
    return Prediction(
        filename=file.filename,
        font=font,
        confidence=round(conf, 3),
        message=None if font else "No text detected.",
    )


@app.get("/health")
def health():
    return {"status": "ok", "classes": len(state.get("classes", [])), "device": DEVICE}


@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    """Predict the font of one image."""
    return _predict_one(file, await file.read())


@app.post("/predict_batch", response_model=BatchResponse)
async def predict_batch(files: List[UploadFile] = File(...)):
    """Predict the font of several images."""
    if not files:
        raise HTTPException(400, "No file provided.")
    preds = [_predict_one(f, await f.read()) for f in files]
    return BatchResponse(predictions=preds, fonts=[p.font for p in preds])
