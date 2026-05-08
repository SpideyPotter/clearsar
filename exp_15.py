"""
ClearSAR — exp_15: YOLO11l P2 + CBAM on the P2/4 branch

Trains the same graph class as the strong baseline (YOLO11-style backbone P1→P5,
extended neck: extra upsample + concat with backbone P2, C3k2 → stride-4 P2 branch,
then downsample path → P3/P4/P5, four-scale Ultralytics Detect), with **CBAM**
inserted on the fused P2 C3k2 output (the tensor fed into Detect at the finest scale).

Motivation (ClearSAR): tiny RFI boxes (e.g. median height ~10 px); P2 gives stride-4
vs P3 stride 8. CBAM refines high-res features before detection.

Dataset: default `./clearsar_yolo/` + a `*.yaml` (e.g. `clearsar.yaml`).

Usage:
    python exp_15.py
    python exp_15.py --data clearsar_yolo --name yolo11l_p2_exp15_cbam
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Paths & training defaults (aligned with in-repo yolo11l_p2_fold0 args.yaml)
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).resolve().parent
MODEL_YAML   = SCRIPT_DIR / "cfg" / "yolo11l-p2.yaml"
PRETRAINED   = "yolo11l.pt"

DATASET_DIR  = SCRIPT_DIR / "clearsar_yolo"
PROJECT      = SCRIPT_DIR / "runs" / "clearsar"
RUN_NAME     = "yolo11l_p2_exp15_cbam"

EPOCHS       = 200
PATIENCE     = 60
BATCH        = 4
IMGSZ        = 1280
WORKERS      = 8
LR0          = 1e-3
LRF          = 0.01
WEIGHT_DECAY = 5e-4
WARMUP_EPOCHS = 5
CLOSE_MOSAIC = 30
DEVICE       = 0 if torch.cuda.is_available() else "cpu"

# Index in `model.model` of the C3k2 block that outputs the P2/4 branch (see cfg/yolo11l-p2.yaml).
P2_C3K2_LAYER_IDX = 19
CBAM_REDUCTION = 16


def find_data_yaml(dataset_dir: Path) -> Path:
    yamls = list(dataset_dir.glob("*.yaml"))
    if not yamls:
        raise FileNotFoundError(f"No .yaml in {dataset_dir}")
    if len(yamls) > 1:
        print(f"  Multiple YAMLs; using {yamls[0].name}")
    return yamls[0]


# ──────────────────────────────────────────────────────────────────────────────
# CBAM
# ──────────────────────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid = max(in_channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(
            self.conv(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1))
        )


class CBAM(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        return x * self.sa(x)


class C3k2CBAMWrapper(nn.Module):
    """Runs the P2 C3k2 block then CBAM (same I/O shapes for downstream Detect)."""

    def __init__(self, inner: nn.Module, in_channels: int, reduction: int):
        super().__init__()
        self.inner = inner
        self.cbam  = CBAM(in_channels, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cbam(self.inner(x))


def _infer_channels_through_layer(model: nn.Module, idx: int, imgsz: int = 256) -> int:
    """Channel count of the tensor produced by `model.model[idx]` (inclusive)."""
    seq = model.model
    dev = next(seq.parameters()).device
    dt  = next(seq.parameters()).dtype
    x = torch.zeros(1, 3, imgsz, imgsz, device=dev, dtype=dt)
    with torch.no_grad():
        for i in range(idx + 1):
            x = seq[i](x)
    return int(x.shape[1])


def inject_cbam_on_p2_branch(model: nn.Module, layer_idx: int = P2_C3K2_LAYER_IDX) -> None:
    """Wrap the P2 C3k2 module with CBAM; mutates `model.model` in place."""
    seq = model.model
    inner = seq[layer_idx]
    ch = _infer_channels_through_layer(model, layer_idx)
    wrap = C3k2CBAMWrapper(inner, ch, CBAM_REDUCTION).to(
        device=next(seq.parameters()).device,
        dtype=next(seq.parameters()).dtype,
    )
    seq[layer_idx] = wrap
    print(f"  CBAM injected after P2 C3k2 (model.model[{layer_idx}], {ch} ch)")


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

def build_trainer_class():
    from ultralytics.models.yolo.detect import DetectionTrainer

    class Exp15Trainer(DetectionTrainer):
        def build_model(self, cfg=None, weights=None, verbose=True):
            model = super().build_model(cfg, weights, verbose)
            inject_cbam_on_p2_branch(model, P2_C3K2_LAYER_IDX)
            return model

    return Exp15Trainer


def parse_args():
    p = argparse.ArgumentParser(description="exp_15 — YOLO11l P2 + CBAM (ClearSAR)")
    p.add_argument("--data", type=str, default=str(DATASET_DIR), help="Dataset root (YOLO layout)")
    p.add_argument("--name", type=str, default=RUN_NAME, help="Run name under project/")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--imgsz", type=int, default=IMGSZ)
    p.add_argument("--workers", type=int, default=WORKERS)
    p.add_argument("--project", type=str, default=str(PROJECT))
    p.add_argument("--device", default=DEVICE, help="0, cpu, 0,1, …")
    p.add_argument("--no-cbam", action="store_true", help="Train baseline P2 graph without CBAM")
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data).resolve()
    if not MODEL_YAML.is_file():
        raise FileNotFoundError(f"Missing model cfg: {MODEL_YAML}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")

    yaml_path = find_data_yaml(data_root)
    print("exp_15 — YOLO11l P2 + CBAM on P2 branch")
    print(f"  model   : {MODEL_YAML}")
    print(f"  weights : {PRETRAINED}")
    print(f"  data    : {yaml_path}")
    print(f"  project : {args.project}  name: {args.name}")

    from ultralytics import YOLO

    model = YOLO(str(MODEL_YAML))
    model.load(PRETRAINED)

    Exp15Trainer = build_trainer_class()
    trainer_kw = {}
    if not args.no_cbam:
        trainer_kw["trainer"] = Exp15Trainer
    else:
        print("  (--no-cbam) using default DetectionTrainer")

    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        patience=PATIENCE,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        optimizer="AdamW",
        lr0=LR0,
        lrf=LRF,
        warmup_epochs=WARMUP_EPOCHS,
        cos_lr=True,
        weight_decay=WEIGHT_DECAY,
        close_mosaic=CLOSE_MOSAIC,
        augment=True,
        mosaic=1.0,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        fliplr=0.5,
        mixup=0.15,
        copy_paste=0.3,
        copy_paste_mode="flip",
        auto_augment="randaugment",
        erasing=0.4,
        project=args.project,
        name=args.name,
        exist_ok=True,
        pretrained=True,
        verbose=True,
        plots=True,
        save=True,
        cache="ram",
        amp=True,
        **trainer_kw,
    )


if __name__ == "__main__":
    main()
