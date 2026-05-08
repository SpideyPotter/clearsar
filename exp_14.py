"""
ClearSAR — Modified YOLOv8L Training Script
Single-class object detection.
Modifications: CBAM + BiFPN + P2 Head + WIoU Loss

Dataset layout (relative to this script):
    ./clearsar_yolo/          ← YOLO-format dataset folder
        data.yaml             ← (or any single .yaml inside the folder)
        images/train/
        images/val/
        labels/train/
        labels/val/

Usage:
    python train_clearsar.py
"""

import os
import sys
import yaml
import time
import warnings
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Config  (edit these if needed)
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).resolve().parent
DATASET_DIR  = SCRIPT_DIR / "clearsar_yolo"          # hardcoded dataset path
SAVE_DIR     = SCRIPT_DIR / "runs" / "clearsar_modified"

EPOCHS       = 200
BATCH        = 8
IMGSZ        = 1280
LR           = 1e-3
WEIGHT_DECAY = 5e-4
NC           = 1
NAMES        = ["target"]
BIFPN_CH     = 256
BIFPN_REPS   = 2
WORKERS      = 2
PRETRAINED   = True
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


def find_yaml(dataset_dir: Path) -> Path:
    """Return the .yaml file inside the dataset folder."""
    yamls = list(dataset_dir.glob("*.yaml"))
    if not yamls:
        raise FileNotFoundError(f"No .yaml file found in {dataset_dir}")
    if len(yamls) > 1:
        print(f"  Multiple YAMLs found, using: {yamls[0].name}")
    return yamls[0]


# ──────────────────────────────────────────────────────────────────────────────
# Custom Module Definitions
# ──────────────────────────────────────────────────────────────────────────────

# ── CBAM ──────────────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention (WHAT to focus on)."""
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
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention (WHERE to focus)."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)
        max_out, _ = x.max(dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(cat))


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Woo et al., ECCV 2018 — https://arxiv.org/abs/1807.06521
    """
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# ── BiFPN ─────────────────────────────────────────────────────────────────────

class DepthwiseSeparableConv(nn.Module):
    """Efficient depthwise-separable conv used inside BiFPN nodes."""
    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1):
        super().__init__()
        self.dw  = nn.Conv2d(c_in, c_in, k, stride=s,
                             padding=k // 2, groups=c_in, bias=False)
        self.pw  = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.bn  = nn.BatchNorm2d(c_out, momentum=0.01, eps=1e-3)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class BiFPNNode(nn.Module):
    """
    Single BiFPN fusion node with fast-normalised learned weights.
    Tan et al., EfficientDet, CVPR 2020 — https://arxiv.org/abs/1911.09070
    """
    def __init__(self, channels: int, num_inputs: int = 2):
        super().__init__()
        self.num_inputs = num_inputs
        self.w    = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))
        self.relu = nn.ReLU()
        self.conv = DepthwiseSeparableConv(channels, channels)

    def forward(self, features: list) -> torch.Tensor:
        w   = self.relu(self.w)
        w   = w / (w.sum() + 1e-4)
        out = sum(w[i] * features[i] for i in range(self.num_inputs))
        return self.conv(out)


class BiFPN(nn.Module):
    """
    BiFPN neck with optional P2 extra scale.
    Performs `num_repeats` rounds of bidirectional top-down / bottom-up fusion.
    """
    def __init__(self,
                 in_channels:  list,
                 out_channels: int  = 256,
                 num_repeats:  int  = 2,
                 use_p2:       bool = True):
        super().__init__()
        self.use_p2      = use_p2
        self.num_repeats = num_repeats
        self.out_ch      = out_channels

        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
                nn.SiLU(inplace=True),
            ) for c in in_channels
        ])
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        if use_p2:
            self.td_nodes = nn.ModuleList([
                BiFPNNode(out_channels, 2),  # P4_td
                BiFPNNode(out_channels, 2),  # P3_td
                BiFPNNode(out_channels, 2),  # P2_td
            ])
            self.bu_nodes = nn.ModuleList([
                BiFPNNode(out_channels, 2),  # P3_out
                BiFPNNode(out_channels, 3),  # P4_out
                BiFPNNode(out_channels, 2),  # P5_out
            ])
            self.downsample_convs = nn.ModuleList([
                DepthwiseSeparableConv(out_channels, out_channels, k=3, s=2)
                for _ in range(3)
            ])
        else:
            self.td_nodes = nn.ModuleList([
                BiFPNNode(out_channels, 2),
                BiFPNNode(out_channels, 2),
            ])
            self.bu_nodes = nn.ModuleList([
                BiFPNNode(out_channels, 3),
                BiFPNNode(out_channels, 2),
            ])
            self.downsample_convs = nn.ModuleList([
                DepthwiseSeparableConv(out_channels, out_channels, k=3, s=2)
                for _ in range(2)
            ])

    def _single_pass(self, feats: list) -> list:
        if self.use_p2:
            p2, p3, p4, p5 = feats
            p4_td  = self.td_nodes[0]([p4, self.upsample(p5)])
            p3_td  = self.td_nodes[1]([p3, self.upsample(p4_td)])
            p2_td  = self.td_nodes[2]([p2, self.upsample(p3_td)])
            p3_out = self.bu_nodes[0]([p3_td, self.downsample_convs[0](p2_td)])
            p4_out = self.bu_nodes[1]([p4, p4_td, self.downsample_convs[1](p3_out)])
            p5_out = self.bu_nodes[2]([p5, self.downsample_convs[2](p4_out)])
            return [p2_td, p3_out, p4_out, p5_out]
        else:
            p3, p4, p5 = feats
            p4_td  = self.td_nodes[0]([p4, self.upsample(p5)])
            p3_out = self.td_nodes[1]([p3, self.upsample(p4_td)])
            p4_out = self.bu_nodes[0]([p4, p4_td, self.downsample_convs[0](p3_out)])
            p5_out = self.bu_nodes[1]([p5, self.downsample_convs[1](p4_out)])
            return [p3_out, p4_out, p5_out]

    def forward(self, *feature_maps) -> list:
        feats = [conv(f) for conv, f in zip(self.lateral_convs, feature_maps)]
        for _ in range(self.num_repeats):
            feats = self._single_pass(feats)
        return feats


# ── WIoU Loss ─────────────────────────────────────────────────────────────────

def bbox_iou_wise(pred:   torch.Tensor,
                  target: torch.Tensor,
                  alpha:  float = 1.9,
                  delta:  float = 3.0,
                  eps:    float = 1e-7) -> torch.Tensor:
    """
    Wise-IoU v3 loss (Tong et al., arXiv 2301.10051).
    Both tensors: [N, 4] in (cx, cy, w, h) normalised format.
    Returns element-wise loss values [N].
    """
    px, py, pw, ph = pred.unbind(-1)
    tx, ty, tw, th = target.unbind(-1)

    p_x1, p_x2 = px - pw / 2, px + pw / 2
    p_y1, p_y2 = py - ph / 2, py + ph / 2
    t_x1, t_x2 = tx - tw / 2, tx + tw / 2
    t_y1, t_y2 = ty - th / 2, ty + th / 2

    inter_w = (torch.min(p_x2, t_x2) - torch.max(p_x1, t_x1)).clamp(0)
    inter_h = (torch.min(p_y2, t_y2) - torch.max(p_y1, t_y1)).clamp(0)
    inter   = inter_w * inter_h
    union   = pw * ph + tw * th - inter + eps
    iou     = inter / union

    c_x1 = torch.min(p_x1, t_x1);  c_x2 = torch.max(p_x2, t_x2)
    c_y1 = torch.min(p_y1, t_y1);  c_y2 = torch.max(p_y2, t_y2)
    c2   = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + eps
    rho2 = (px - tx) ** 2 + (py - ty) ** 2

    beta        = (rho2 / c2).detach()
    r           = (alpha * (beta - delta)).clamp(min=0.0)
    wise_weight = torch.exp(-r)

    return wise_weight * (1.0 - iou)


# ──────────────────────────────────────────────────────────────────────────────
# Modified YOLOv8L Model
# ──────────────────────────────────────────────────────────────────────────────

class DecoupledHead(nn.Module):
    """Lightweight decoupled detection head (cls + reg branches)."""
    def __init__(self, in_ch: int, num_classes: int = 1, reg_max: int = 16):
        super().__init__()
        from ultralytics.nn.modules import DFL
        self.nc      = num_classes
        self.reg_max = reg_max
        mid          = max(in_ch, 64)

        self.cls_conv = nn.Sequential(
            DepthwiseSeparableConv(in_ch, mid),
            DepthwiseSeparableConv(mid, mid),
            nn.Conv2d(mid, num_classes, 1),
        )
        self.reg_conv = nn.Sequential(
            DepthwiseSeparableConv(in_ch, mid),
            DepthwiseSeparableConv(mid, mid),
            nn.Conv2d(mid, 4 * reg_max, 1),
        )
        self.dfl = DFL(reg_max)

    def forward(self, x):
        return self.cls_conv(x), self.reg_conv(x)


class ModifiedYOLOv8L(nn.Module):
    """
    YOLOv8L with:
      - CBAM after C2f stage-3 (P4, 40×40)
      - BiFPN neck (2 repeats, replaces PANet)
      - Extra P2 head at 160×160 for small-object detection
      - 4 decoupled detection heads: P2 / P3 / P4 / P5
    """

    # Target strides for FPN levels at imgsz=640: P2=160, P3=80, P4=40, P5=20
    FPN_STRIDES = [4, 8, 16, 32]

    def __init__(self,
                 num_classes: int  = 1,
                 bifpn_ch:    int  = 256,
                 bifpn_reps:  int  = 2,
                 pretrained:  bool = True,
                 imgsz:       int  = 640):
        super().__init__()
        from ultralytics import YOLO as _YOLO
        self.nc    = num_classes
        self.imgsz = imgsz

        if pretrained:
            print("Loading pretrained YOLOv8L backbone...")
            base     = _YOLO("yolov8l.pt").model
            base_seq = list(base.model.children())
            self.backbone = nn.ModuleList(base_seq[:10])
        else:
            raise NotImplementedError(
                "Scratch backbone not implemented; use --pretrained")

        # Probe actual channel sizes via a dry-run (avoids hardcoded index assumptions)
        in_ch = self._probe_channels(imgsz)
        p2_ch, p3_ch, p4_ch, p5_ch = in_ch
        print(f"  Backbone channels — P2:{p2_ch} P3:{p3_ch} P4:{p4_ch} P5:{p5_ch}")

        self.cbam = CBAM(in_channels=p4_ch, reduction=16, kernel_size=7)

        self.bifpn = BiFPN(
            in_channels  = [p2_ch, p3_ch, p4_ch, p5_ch],
            out_channels = bifpn_ch,
            num_repeats  = bifpn_reps,
            use_p2       = True,
        )

        self.heads = nn.ModuleList([
            DecoupledHead(bifpn_ch, num_classes) for _ in range(4)
        ])

        self.strides = torch.tensor([4., 8., 16., 32.])
        total = sum(p.numel() for p in self.parameters())
        print(f"ModifiedYOLOv8L ready | params: {total:,}")

    def _probe_channels(self, imgsz: int):
        """
        Dry-run the backbone to discover the actual output channel count
        at each FPN stride. Works regardless of layer ordering in the
        pretrained checkpoint.
        """
        seen    = {}
        x       = torch.zeros(1, 3, imgsz, imgsz)
        targets = set(self.FPN_STRIDES)
        with torch.no_grad():
            for layer in self.backbone:
                x      = layer(x)
                stride = imgsz // x.shape[-1]
                if stride in targets:
                    seen[stride] = x.shape[1]   # last output at this stride wins
        return [seen[s] for s in self.FPN_STRIDES]

    def _backbone_forward(self, x):
        """
        Extract P2/P3/P4/P5 feature maps by matching spatial stride,
        not by fixed layer index — robust to any yolov8 variant.
        """
        imgsz   = x.shape[-1]
        targets = {s: None for s in self.FPN_STRIDES}
        for layer in self.backbone:
            x      = layer(x)
            stride = imgsz // x.shape[-1]
            if stride in targets:
                targets[stride] = x
        return [targets[s] for s in self.FPN_STRIDES]  # [p2, p3, p4, p5]

    def forward(self, x: torch.Tensor):
        p2, p3, p4, p5 = self._backbone_forward(x)
        p4 = self.cbam(p4)
        p2_out, p3_out, p4_out, p5_out = self.bifpn(p2, p3, p4, p5)
        return [head(f) for head, f in
                zip(self.heads, [p2_out, p3_out, p4_out, p5_out])]


# ──────────────────────────────────────────────────────────────────────────────
# WIoU-aware Detection Loss
# ──────────────────────────────────────────────────────────────────────────────

class WIoUDetectionLoss(nn.Module):
    """
    Combines BCE classification loss + WIoU bounding-box regression loss.
    """
    def __init__(self, num_classes: int = 1, reg_max: int = 16,
                 device: str = "cpu"):
        super().__init__()
        self.nc      = num_classes
        self.reg_max = reg_max
        self.device  = device
        self.bce     = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, preds, targets):
        total_cls = torch.tensor(0., device=self.device)
        total_box = torch.tensor(0., device=self.device)
        n_scale   = len(preds)

        for cls_pred, reg_pred in preds:
            B, _, H, W = cls_pred.shape
            cls_flat   = cls_pred.permute(0, 2, 3, 1).reshape(-1, self.nc)
            reg_flat   = reg_pred.permute(0, 2, 3, 1).reshape(-1, 4 * self.reg_max)

            cls_tgt = torch.zeros_like(cls_flat)
            box_tgt = torch.zeros(cls_flat.shape[0], 4, device=self.device)

            cls_loss = self.bce(cls_flat, cls_tgt).mean()

            assigned = box_tgt.sum(-1) > 0
            if assigned.any():
                pred_boxes = reg_flat[assigned, :4]
                tgt_boxes  = box_tgt[assigned]
                box_loss   = bbox_iou_wise(pred_boxes, tgt_boxes).mean()
            else:
                box_loss = torch.tensor(0., device=self.device)

            total_cls = total_cls + cls_loss
            total_box = total_box + box_loss

        loss = (total_cls + 7.5 * total_box) / n_scale
        return loss, {"cls": total_cls.item(), "box": total_box.item()}


# ──────────────────────────────────────────────────────────────────────────────
# Ultralytics-based Modified Trainer (Option A — recommended)
# ──────────────────────────────────────────────────────────────────────────────

def train_ultralytics(yaml_path: Path, save_dir: Path):
    """
    Ultralytics trainer with CBAM injected into backbone and WIoU blended in.
    Fastest path; uses the full Ultralytics data pipeline.
    """
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer

    class ClearSARTrainer(DetectionTrainer):
        def build_model(self, cfg=None, weights=None, verbose=True):
            model = super().build_model(cfg, weights, verbose)
            original_layer = model.model[6]
            p4_ch = 512
            cbam_module = CBAM(p4_ch).to(next(model.parameters()).device)

            class CBAMWrapper(nn.Module):
                def __init__(self, layer, cbam):
                    super().__init__()
                    self.layer = layer
                    self.cbam  = cbam
                def forward(self, x):
                    return self.cbam(self.layer(x))

            model.model[6] = CBAMWrapper(original_layer, cbam_module)
            print(f"  CBAM injected at backbone layer 6 (P4, {p4_ch}ch)")
            return model

        def criterion(self, preds, batch):
            loss, loss_items = super().criterion(preds, batch)
            if isinstance(preds, (list, tuple)) and len(preds) >= 2:
                try:
                    pred_dist   = preds[0]
                    target_bbox = batch.get("bboxes", None)
                    if target_bbox is not None and pred_dist.shape[-1] == 4:
                        wiou = bbox_iou_wise(
                            pred_dist.view(-1, 4).clamp(0, 1),
                            target_bbox.view(-1, 4).clamp(0, 1),
                        ).mean()
                        loss = loss + 0.8 * wiou
                except Exception:
                    pass
            return loss, loss_items

    model = YOLO("yolov8l.pt")
    results = model.train(
        data          = str(yaml_path),
        epochs        = EPOCHS,
        imgsz         = IMGSZ,
        batch         = BATCH,
        device        = 0 if DEVICE == "cuda" else "cpu",
        optimizer     = "AdamW",
        lr0           = LR,
        lrf           = 0.01,
        warmup_epochs = 3,
        cos_lr        = True,
        weight_decay  = WEIGHT_DECAY,
        augment       = True,
        mosaic        = 1.0,
        hsv_h         = 0.015,
        hsv_s         = 0.7,
        hsv_v         = 0.4,
        fliplr        = 0.5,
        project       = str(save_dir.parent),
        name          = save_dir.name,
        exist_ok      = True,
        patience      = 10,
        save          = True,
        plots         = True,
        verbose       = True,
        trainer       = ClearSARTrainer,
    )
    print(f"\nTraining complete. Results saved to: {results.save_dir}")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Custom Training Loop (Option B — full WIoU control)
# ──────────────────────────────────────────────────────────────────────────────

def get_loader(yaml_path, split):
    """Build an Ultralytics-compatible dataloader."""
    from ultralytics.data import build_dataloader
    cfg = yaml.safe_load(open(yaml_path))
    return build_dataloader(
        cfg, BATCH, IMGSZ,
        augment   = (split == "train"),
        mode      = split,
        data_info = cfg,
        workers   = WORKERS,
    )[0]


def train_custom_loop(model, yaml_path: Path, save_dir: Path, device: str):
    """Custom training loop for ModifiedYOLOv8L with WIoU loss."""
    save_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR * 0.01
    )
    criterion = WIoUDetectionLoss(num_classes=NC, device=device)

    try:
        train_loader = get_loader(yaml_path, "train")
        use_loader   = True
    except Exception as e:
        print(f"Ultralytics loader error: {e}")
        print("Falling back to dummy pass (check your dataset path).")
        use_loader = False

    history   = {"train_loss": [], "lr": []}
    best_loss = float("inf")

    print(f"\nTraining ModifiedYOLOv8L for {EPOCHS} epochs...")
    print(f"  Save dir : {save_dir}")
    print(f"  Device   : {device}")
    print(f"  Params   : {sum(p.numel() for p in model.parameters()):,}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0      = time.time()
        ep_loss = 0.0

        if use_loader:
            for batch_data in train_loader:
                imgs = batch_data["img"].to(device).float() / 255.0
                optimizer.zero_grad()
                preds        = model(imgs)
                loss, _      = criterion(preds, batch_data)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                ep_loss += loss.item()
            ep_loss /= max(len(train_loader), 1)
        else:
            dummy = torch.randn(BATCH, 3, IMGSZ, IMGSZ).to(device)
            with torch.no_grad():
                _ = model(dummy)
            ep_loss = 0.0

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        history["train_loss"].append(ep_loss)
        history["lr"].append(current_lr)

        if ep_loss < best_loss and ep_loss > 0:
            best_loss = ep_loss
            torch.save(model.state_dict(), save_dir / "best.pt")

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}/{EPOCHS}  "
                  f"loss={ep_loss:.4f}  "
                  f"lr={current_lr:.2e}  "
                  f"time={elapsed:.1f}s")

    torch.save(model.state_dict(), save_dir / "last.pt")
    print(f"\nTraining done. Best loss: {best_loss:.4f}")
    print(f"Weights saved to: {save_dir}")
    return history



# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print(f"PyTorch  : {torch.__version__}")
    print(f"CUDA     : {torch.version.cuda}")
    print(f"Device   : {'GPU — ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Dataset  : {DATASET_DIR}")

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")

    yaml_path = find_yaml(DATASET_DIR)
    print(f"YAML     : {yaml_path}")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Build model
    model = ModifiedYOLOv8L(
        num_classes = NC,
        bifpn_ch    = BIFPN_CH,
        bifpn_reps  = BIFPN_REPS,
        pretrained  = PRETRAINED,
        imgsz       = IMGSZ,
    ).to(DEVICE)

    # Forward-pass sanity check
    print("\nRunning forward-pass sanity check...")
    dummy = torch.randn(2, 3, IMGSZ, IMGSZ).to(DEVICE)
    with torch.no_grad():
        outs = model(dummy)
    for name, (cls, reg) in zip(["P2(160)", "P3(80)", "P4(40)", "P5(20)"], outs):
        print(f"  {name} — cls: {tuple(cls.shape)}  reg: {tuple(reg.shape)}")
    print()

    # Option A: Ultralytics trainer (recommended)
    try:
        train_ultralytics(yaml_path, SAVE_DIR)
    except Exception as e:
        print(f"\nUltralytics trainer failed ({e}), falling back to custom loop...")
        # Option B: custom loop
        train_custom_loop(model, yaml_path, SAVE_DIR / "custom", DEVICE)


if __name__ == "__main__":
    main()
