"""Font Classifier: training and evaluation.

1. Load a folder-per-font dataset (label = folder name).
2. Apply the same preprocessing as inference: HSV value + resize-with-padding.
3. Fine-tune a ResNet18 (ImageNet) with augmentations that mimic real photos.
4. Optionally evaluate on real images with the full inference pipeline
   (EasyOCR detection on upscale x2 + HSV, word crops, confidence-weighted vote).
5. Save the model, training curves, confusion matrix and metrics.

Usage:
    python train.py --train-dir data/train --out models/
    python train.py --train-dir data/train --test-dir data/real --out models/

Test images are labelled either by their parent folder name or by a "font"
entry in the PNG text metadata.
"""
import argparse
import collections
import io
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

from src.preprocessing import resize_pad, to_hsv_value, upscale

NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class RandomJPEG:
    """Random JPEG compression, to simulate the artifacts of real images."""

    def __init__(self, p=0.5, qmin=25, qmax=75):
        self.p, self.qmin, self.qmax = p, qmin, qmax

    def __call__(self, img):
        if random.random() < self.p:
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=random.randint(self.qmin, self.qmax))
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        return img


def add_noise(t):
    return t + 0.05 * torch.randn_like(t)


train_tf = transforms.Compose([
    transforms.Lambda(to_hsv_value),
    transforms.Lambda(resize_pad),
    transforms.RandomApply([transforms.RandomAffine(5, scale=(0.5, 1.0), shear=4, fill=255)], p=0.7),
    transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.5),
    RandomJPEG(p=0.5),
    transforms.ToTensor(),
    transforms.RandomApply([transforms.Lambda(add_noise)], p=0.3),
    NORM,
])

eval_tf = transforms.Compose([
    transforms.Lambda(to_hsv_value),
    transforms.Lambda(resize_pad),
    transforms.ToTensor(),
    NORM,
])


def stratified_split(samples, ratio=0.8):
    by_class = collections.defaultdict(list)
    for idx, (_, label) in enumerate(samples):
        by_class[label].append(idx)
    train_idx, val_idx = [], []
    for idxs in by_class.values():
        random.shuffle(idxs)
        cut = int(ratio * len(idxs))
        train_idx += idxs[:cut]
        val_idx += idxs[cut:]
    return train_idx, val_idx


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total = correct = 0
    loss_sum = 0.0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total


def predict_proba(model, reader, pil_img, n_classes, device):
    """Aggregated probability vector over all detected words, or None."""
    det = to_hsv_value(upscale(pil_img, 2))
    results = reader.readtext(np.array(det), text_threshold=0.3, low_text=0.3)
    probs = torch.zeros(n_classes, device=device)
    n = 0
    for bbox, text, conf in results:
        if len(text.strip()) < 2:
            continue
        xs = [p[0] / 2 for p in bbox]
        ys = [p[1] / 2 for p in bbox]
        crop = pil_img.crop((max(int(min(xs)), 0), max(int(min(ys)), 0),
                             int(max(xs)), int(max(ys))))
        if crop.width < 3 or crop.height < 3:
            continue
        x = eval_tf(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            probs += conf * F.softmax(model(x)[0], dim=0)
        n += 1
    return (probs / probs.sum()).cpu().numpy() if n else None


def label_of(path: Path, img: Image.Image) -> str:
    return getattr(img, "text", {}).get("font") or path.parent.name


def evaluate(model, classes, test_dir, out, device):
    import easyocr
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        top_k_accuracy_score,
    )

    reader = easyocr.Reader(["en"], gpu=(device == "cuda"))
    model.eval()
    paths = [p for p in Path(test_dir).rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    y_true, y_proba, latencies, no_detect = [], [], [], 0
    for p in paths:
        with Image.open(p) as im:
            im.load()
            font = label_of(p, im)
            img = im.convert("RGB")
        if font not in classes:
            continue
        t0 = time.time()
        proba = predict_proba(model, reader, img, len(classes), device)
        latencies.append(time.time() - t0)
        if proba is None:  # no text detected: uniform, counts as an error
            no_detect += 1
            proba = np.ones(len(classes)) / len(classes)
        y_true.append(classes.index(font))
        y_proba.append(proba)

    y_true, y_proba = np.array(y_true), np.array(y_proba)
    y_pred = y_proba.argmax(1)
    labels = range(len(classes))
    metrics = {
        "n_images": len(y_true),
        "no_detect": no_detect,
        "top1": accuracy_score(y_true, y_pred),
        "top3": top_k_accuracy_score(y_true, y_proba, k=3, labels=labels),
        "top5": top_k_accuracy_score(y_true, y_proba, k=5, labels=labels),
        "latency_ms": float(np.mean(latencies) * 1000),
    }
    print(json.dumps(metrics, indent=2))

    present = sorted(set(y_true) | set(y_pred))
    names = [classes[i] for i in present]
    print(classification_report(y_true, y_pred, labels=present, target_names=names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=present)
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(names)), names, rotation=90, fontsize=7)
    plt.yticks(range(len(names)), names, fontsize=7)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion matrix (top-1 = {metrics['top1']:.1%})")
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j]:
                plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close()
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", type=Path, required=True)
    p.add_argument("--test-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("models"))
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: no GPU found, training will be slow.")

    full_train = datasets.ImageFolder(str(args.train_dir), transform=train_tf)
    full_eval = datasets.ImageFolder(str(args.train_dir), transform=eval_tf)
    classes = full_train.classes
    train_idx, val_idx = stratified_split(full_train.samples)
    print(f"{len(classes)} classes | train {len(train_idx)} | val {len(val_idx)}")

    loader_kw = dict(batch_size=args.batch_size, num_workers=2, pin_memory=(device == "cuda"))
    train_dl = DataLoader(Subset(full_train, train_idx), shuffle=True, **loader_kw)
    val_dl = DataLoader(Subset(full_eval, val_idx), shuffle=False, **loader_kw)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        _, tr_acc = run_epoch(model, train_dl, criterion, device, optimizer)
        _, va_acc = run_epoch(model, val_dl, criterion, device)
        scheduler.step()
        history.append({"epoch": epoch, "train_acc": tr_acc, "val_acc": va_acc})
        print(f"epoch {epoch} | train {tr_acc:.3f} | val {va_acc:.3f} | {time.time() - t0:.0f}s")

    torch.save({"model": model.state_dict(), "classes": classes}, args.out / "resnet18_final.pt")

    epochs = [h["epoch"] for h in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [h["train_acc"] for h in history], "o-", label="train")
    plt.plot(epochs, [h["val_acc"] for h in history], "s-", label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out / "training_curves.png", dpi=150)
    plt.close()

    metrics = {"history": history}
    if args.test_dir:
        metrics.update(evaluate(model, classes, args.test_dir, args.out, device))
    with open(args.out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Artifacts saved in {args.out}")


if __name__ == "__main__":
    main()
