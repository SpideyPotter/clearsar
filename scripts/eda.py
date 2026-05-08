"""ClearSAR EDA — scans the YOLO-format dataset and saves plots + a stats JSON.

Run:
    python scripts/eda.py

Outputs:
    eda/stats.json
    eda/boxes_per_image.png
    eda/box_size_hist.png
    eda/aspect_ratio_hist.png
    eda/spatial_heatmap.png
    eda/image_size_hist.png
"""

from __future__ import annotations

import glob
import json
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "clearsar_yolo"
OUT = ROOT / "eda"
OUT.mkdir(exist_ok=True)

SPLITS = ["train", "val"]
IMAGE_SIZE_SPLITS = ["train", "val", "test"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# COCO-style small/medium/large thresholds (area in pixels, at native image size)
COCO_SMALL = 32 * 32
COCO_MEDIUM = 96 * 96


def scan_split(split: str) -> dict:
    img_dir = DATA / "images" / split
    lbl_dir = DATA / "labels" / split
    if not img_dir.exists():
        return {}

    images = sorted(
        str(p)
        for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    labels = sorted(glob.glob(str(lbl_dir / "*.txt")))

    img_sizes, boxes_per_image, ws, hs, cxs, cys, areas = [], [], [], [], [], [], []
    backgrounds = 0

    for img_path in images:
        stem = Path(img_path).stem
        lbl_path = lbl_dir / f"{stem}.txt"
        with Image.open(img_path) as im:
            W, H = im.size
        img_sizes.append((W, H))

        if not lbl_path.exists() or lbl_path.stat().st_size == 0:
            backgrounds += 1
            boxes_per_image.append(0)
            continue

        n = 0
        with open(lbl_path) as f:
            for ln in f:
                parts = ln.split()
                if len(parts) < 5:
                    continue
                _, cx, cy, w, h = (float(x) for x in parts[:5])
                cxs.append(cx)
                cys.append(cy)
                ws.append(w * W)
                hs.append(h * H)
                areas.append(w * W * h * H)
                n += 1
        boxes_per_image.append(n)

    ws_a = np.array(ws)
    hs_a = np.array(hs)
    areas_a = np.array(areas)

    return {
        "split": split,
        "num_images": len(images),
        "num_labels_files": len(labels),
        "backgrounds": backgrounds,
        "num_boxes": len(ws),
        "image_sizes_top5": [
            {"size": f"{w}x{h}", "count": c}
            for (w, h), c in Counter(img_sizes).most_common(5)
        ],
        "boxes_per_image": {
            "mean": float(np.mean(boxes_per_image)) if boxes_per_image else 0.0,
            "median": float(np.median(boxes_per_image)) if boxes_per_image else 0.0,
            "min": int(np.min(boxes_per_image)) if boxes_per_image else 0,
            "max": int(np.max(boxes_per_image)) if boxes_per_image else 0,
        },
        "width_px": _stats(ws_a),
        "height_px": _stats(hs_a),
        "area_px": _stats(areas_a),
        "sqrt_wh_px": _stats(np.sqrt(ws_a * hs_a)) if len(ws_a) else {},
        "aspect_ratio_w_over_h": _stats(ws_a / np.maximum(hs_a, 1e-6)) if len(ws_a) else {},
        "coco_size_bucket": {
            "small (area<32^2)": int((areas_a < COCO_SMALL).sum()),
            "medium (32^2..96^2)": int(((areas_a >= COCO_SMALL) & (areas_a < COCO_MEDIUM)).sum()),
            "large (>96^2)": int((areas_a >= COCO_MEDIUM).sum()),
        },
        "tiny_boxes_lt_3px": int(((ws_a < 3) | (hs_a < 3)).sum()),
        "center_x_norm": _stats(np.array(cxs)) if cxs else {},
        "center_y_norm": _stats(np.array(cys)) if cys else {},
        # keep raw arrays for plotting (not serialized)
        "_raw": {
            "ws": ws_a, "hs": hs_a, "cxs": np.array(cxs), "cys": np.array(cys),
            "boxes_per_image": np.array(boxes_per_image),
            "img_sizes": img_sizes,
        },
    }


def _stats(a: np.ndarray) -> dict:
    if len(a) == 0:
        return {}
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "min": float(a.min()),
        "max": float(a.max()),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
    }


def plot_all(per_split: dict[str, dict], image_size_split: dict[str, dict]) -> None:
    plt.rcParams["figure.dpi"] = 110

    # 1. Boxes per image
    fig, axes = plt.subplots(1, len(per_split), figsize=(4 * len(per_split), 3), sharey=True)
    for ax, (split, s) in zip(np.atleast_1d(axes), per_split.items()):
        ax.hist(s["_raw"]["boxes_per_image"], bins=30, color="#2b7fbf", edgecolor="white")
        ax.set_title(f"{split} — boxes/image"); ax.set_xlabel("boxes"); ax.set_ylabel("images")
    fig.tight_layout(); fig.savefig(OUT / "boxes_per_image.png"); plt.close(fig)

    # 2. Box size histograms (width vs height, log-y)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for split, s in per_split.items():
        axes[0].hist(s["_raw"]["ws"], bins=40, histtype="step", label=split, linewidth=1.5)
        axes[1].hist(s["_raw"]["hs"], bins=40, histtype="step", label=split, linewidth=1.5)
    for ax, title, xl in zip(axes, ["Box width (px)", "Box height (px)"], ["w", "h"]):
        ax.set_yscale("log"); ax.set_title(title); ax.set_xlabel(xl); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "box_size_hist.png"); plt.close(fig)

    # 3. Aspect ratio (log) histogram
    fig, ax = plt.subplots(figsize=(6, 3))
    for split, s in per_split.items():
        ar = s["_raw"]["ws"] / np.maximum(s["_raw"]["hs"], 1e-6)
        ax.hist(np.log10(np.clip(ar, 1e-2, 1e2)), bins=50, histtype="step", label=split, linewidth=1.5)
    ax.set_xlabel("log10(aspect ratio w/h)"); ax.set_ylabel("count"); ax.set_title("Aspect ratio (log10)")
    ax.axvline(0, color="k", linewidth=0.6); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "aspect_ratio_hist.png"); plt.close(fig)

    # 4. Spatial heatmap of box centers (normalized coords)
    fig, axes = plt.subplots(1, len(per_split), figsize=(4 * len(per_split), 3.6))
    for ax, (split, s) in zip(np.atleast_1d(axes), per_split.items()):
        if len(s["_raw"]["cxs"]):
            h, xe, ye = np.histogram2d(
                s["_raw"]["cxs"], s["_raw"]["cys"], bins=40, range=[[0, 1], [0, 1]])
            ax.imshow(h.T, origin="upper", extent=(0, 1, 1, 0), cmap="viridis", aspect="auto")
        ax.set_title(f"{split} — center density"); ax.set_xlabel("cx"); ax.set_ylabel("cy")
    fig.tight_layout(); fig.savefig(OUT / "spatial_heatmap.png"); plt.close(fig)

    # 5. Image size distribution
    fig, ax = plt.subplots(figsize=(6, 3))
    for split, s in image_size_split.items():
        sizes = s["_raw"]["img_sizes"]
        Ws = [w for w, _ in sizes]; Hs = [h for _, h in sizes]
        ax.scatter(Ws, Hs, alpha=0.3, s=10, label=f"{split} (n={len(sizes)})")
    ax.set_xlabel("width (px)"); ax.set_ylabel("height (px)"); ax.set_title("Image sizes")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "image_size_hist.png"); plt.close(fig)


def main() -> None:
    per_split = {}
    for sp in SPLITS:
        res = scan_split(sp)
        if res:
            per_split[sp] = res

    image_size_split = {}
    for sp in IMAGE_SIZE_SPLITS:
        res = scan_split(sp)
        if res:
            image_size_split[sp] = res

    plot_all(per_split, image_size_split)

    # strip raw arrays before dumping json
    serializable = {k: {kk: vv for kk, vv in v.items() if kk != "_raw"} for k, v in per_split.items()}
    with open(OUT / "stats.json", "w") as f:
        json.dump(serializable, f, indent=2)

    # compact console summary
    for split, s in per_split.items():
        print(f"\n=== {split} ===")
        print(f"  images       : {s['num_images']}  (backgrounds: {s['backgrounds']})")
        print(f"  boxes        : {s['num_boxes']}")
        bpi = s["boxes_per_image"]
        print(f"  boxes/image  : mean={bpi['mean']:.2f}  median={bpi['median']}  max={bpi['max']}")
        if s["num_boxes"]:
            w = s["width_px"]; h = s["height_px"]; a = s["area_px"]
            print(f"  width px     : p10={w['p10']:.1f}  med={w['median']:.1f}  p90={w['p90']:.1f}")
            print(f"  height px    : p10={h['p10']:.1f}  med={h['median']:.1f}  p90={h['p90']:.1f}")
            print(f"  tiny (<3px)  : {s['tiny_boxes_lt_3px']}")
            print(f"  COCO bucket  : {s['coco_size_bucket']}")


if __name__ == "__main__":
    main()
