"""
ClearSAR Experiment 2 -- YOLOv9e Training
PGI + GELAN architecture for ensemble diversity.

Usage:
    conda activate sar
    python train_v9_yolov9e.py

    Kubernetes / low-RAM tuning (optional env):
        CLEARSAR_TRAIN_IMGSZ   default 1280
        CLEARSAR_TRAIN_BATCH   default 4
        CLEARSAR_TRAIN_WORKERS default 16
        CLEARSAR_TRAIN_CACHE   ram | disk | false (default ram)

Monitor:
    tail -f runs/clearsar/yolov9e_exp9/results.csv
"""

import os
import torch
from pathlib import Path
from ultralytics import YOLO


def _train_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v is not None and v != "" else default


def _train_cache():
    """Ultralytics cache: True | False | 'ram' | 'disk' (from CLEARSAR_TRAIN_CACHE)."""
    raw = os.environ.get("CLEARSAR_TRAIN_CACHE")
    if raw is None or raw == "":
        return "ram"
    lowered = raw.strip().lower()
    if lowered in ("false", "0", "no", "none"):
        return False
    if lowered in ("true", "1", "yes"):
        return True
    if lowered in ("ram", "disk"):
        return lowered
    return raw

print("=" * 60)
print("  ClearSAR Exp2 -- YOLOv9e Training")
print("=" * 60)
print(f"  GPU  : {torch.cuda.get_device_name(0)}")
print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
assert torch.cuda.is_available(), "CUDA not available!"

DATA_YAML = "clearsar_yolo/clearsar.yaml"
PROJECT   = "runs/clearsar"
EXP_NAME  = "yolov9e_exp9"
MODEL     = "yolov9e.pt"
EPOCHS    = 200
IMGSZ     = _train_int("CLEARSAR_TRAIN_IMGSZ", 1280)
BATCH     = _train_int("CLEARSAR_TRAIN_BATCH", 4)
WORKERS   = _train_int("CLEARSAR_TRAIN_WORKERS", 16)
CACHE     = _train_cache()
PATIENCE  = 60

print(f"  Model   : {MODEL}")
print(f"  imgsz   : {IMGSZ}")
print(f"  batch   : {BATCH}")
print(f"  cache   : {CACHE!r}")
print(f"  workers : {WORKERS}")
print(f"  epochs  : {EPOCHS}")

model = YOLO(MODEL)

results = model.train(
    data         = DATA_YAML,
    epochs       = EPOCHS,
    imgsz        = IMGSZ,
    batch        = BATCH,
    workers      = WORKERS,
    device       = 0,
    project      = PROJECT,
    name         = EXP_NAME,
    exist_ok     = True,

    optimizer    = "AdamW",
    lr0          = 0.001,
    lrf          = 0.01,
    weight_decay = 0.0005,
    warmup_epochs= 5,
    cos_lr       = True,

    box          = 10.0,
    cls          = 0.5,
    dfl          = 2.0,

    patience     = PATIENCE,

    hsv_h        = 0.015,
    hsv_s        = 0.5,
    hsv_v        = 0.4,
    degrees      = 0.0,
    translate    = 0.1,
    scale        = 0.9,
    shear        = 0.0,
    perspective  = 0.0,
    flipud       = 0.0,
    fliplr       = 0.5,
    mosaic       = 1.0,
    mixup        = 0.15,
    copy_paste   = 0.3,
    erasing      = 0.4,
    close_mosaic = 30,

    amp          = True,
    cache        = CACHE,
    save         = True,
    save_period  = 10,
    plots        = True,
    verbose      = True,
    label_smoothing = 0.0,
)

import glob
candidates = sorted(glob.glob(f"**/{EXP_NAME}/weights/best.pt", recursive=True))
best_path = max(candidates, key=lambda p: Path(p).stat().st_mtime) if candidates else f"{PROJECT}/{EXP_NAME}/weights/best.pt"

print(f"\n  Validating: {best_path}")
best_model = YOLO(best_path)
val = best_model.val(data=DATA_YAML, imgsz=IMGSZ, batch=BATCH, device=0, plots=True)
print(f"\n  mAP@50    : {val.box.map50:.4f}")
print(f"  mAP@50-95 : {val.box.map:.4f}")
