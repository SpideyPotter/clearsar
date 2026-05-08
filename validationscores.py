"""
Print validation mAP (solo, per model) using the same setup as ensemble_predict_v2.py.

Usage:
    conda activate sar
    python val_scores_per_model.py
    python val_scores_per_model.py --5fold-only
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import tempfile
from pathlib import Path

from PIL import Image
from tqdm import tqdm
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Zero area box skipped",
    category=UserWarning,
    module=r"ensemble_boxes\.ensemble_boxes_wbf",
)

REPO_ROOT = Path(__file__).resolve().parent

MODEL_CANDIDATES = [
    ("fold0_p2", "runs/detect/runs/clearsar/yolo11l_p2_fold0/weights/best.pt", 1280),
    ("fold1_p2", "runs/detect/runs/clearsar/yolo11l_p2_fold1/weights/best.pt", 1280),
    ("fold2_p2", "runs/detect/runs/clearsar/yolo11l_p2_fold2/weights/best.pt", 1280),
    ("fold3_p2", "runs/detect/runs/clearsar/yolo11l_p2_fold3/weights/best.pt", 1280),
    ("fold4_p2", "runs/detect/runs/clearsar/yolo11l_p2_fold4/weights/best.pt", 1280),
    ("rtdetr_x", "RT_detr_56.pt", 1280),
    ("yolo11x", "runs/detect/runs/clearsar/yolo11x_exp8/weights/best.pt", 1280),
    ("yolov9e", "runs/detect/runs/clearsar/yolov9e_exp9/weights/best.pt", 1280),
    ("exp4_p2", "runs/detect/runs/clearsar/yolo11l_p2_exp4/weights/best.pt", 1280),
    ("exp5_v8x", "runs/detect/runs/clearsar/yolov8x_exp5/weights/best.pt", 1280),
    ("exp3_y11m", "runs/detect/runs/clearsar/yolo11m_exp3/weights/best.pt", 1280),
    ("exp22_v8l", "runs/detect/runs/clearsar/yolov8l_exp22/weights/best.pt", 800),
    ("exp1_v8m", "runs/detect/runs/clearsar/yolov8m_exp1/weights/best.pt", 640),
]

FIVE_FOLD_NAMES = tuple(f"fold{i}_p2" for i in range(5))


def _resolve(p: str | Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else REPO_ROOT / pp


def read_checkpoint_epoch(weights_path: str | Path) -> str:
    """Training epoch label from Ultralytics checkpoint, or filename fallback (e.g. RT_detr_56.pt)."""
    path = Path(weights_path)
    try:
        import torch

        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict):
            e = ckpt.get("epoch")
            if e is not None:
                # Ultralytics stores last completed epoch index (0-based); display 1-based label.
                return str(int(e) + 1)
    except Exception:
        pass
    m = re.search(r"(\d+)", path.stem)
    if m:
        return m.group(1)
    return "—"


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-model validation mAP (solo)")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou-nms", type=float, default=0.5)
    parser.add_argument("--cache-dir", type=str, default="pred_cache_v2")
    parser.add_argument("--tta", action="store_true", default=True)
    parser.add_argument(
        "--5fold-only",
        action="store_true",
        dest="five_fold_only",
        help="Only evaluate fold0_p2..fold4_p2",
    )
    args = parser.parse_args()

    ann_file = REPO_ROOT / "ClearSAR/data/annotations/instances_train.json"
    val_dir = REPO_ROOT / "clearsar_yolo/images/val"
    val_labels = REPO_ROOT / "clearsar_yolo/labels/val"
    cache_dir = REPO_ROOT / Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    with open(ann_file) as f:
        coco = json.load(f)
    cat_id = coco["categories"][0]["id"]

    if args.five_fold_only:
        candidates = [t for t in MODEL_CANDIDATES if t[0] in FIVE_FOLD_NAMES]
    else:
        candidates = MODEL_CANDIDATES

    available = [(n, str(_resolve(p)), s) for n, p, s in candidates if _resolve(p).is_file()]
    if not available:
        raise SystemExit("No model weights found — check paths under MODEL_CANDIDATES.")

    epoch_by_name = {n: read_checkpoint_epoch(wpath) for n, wpath, _ in available}

    img_paths = sorted(val_dir.glob("*.png"))
    img_sizes: dict[int, tuple[int, int]] = {}
    for p in img_paths:
        img = Image.open(p)
        img_sizes[int(p.stem)] = (img.width, img.height)
        img.close()
    image_ids = sorted(img_sizes.keys())

    def build_gt():
        gt_images, gt_anns = [], []
        ann_id = 1
        for lbl_path in sorted(val_labels.glob("*.txt")):
            image_id = int(lbl_path.stem)
            if image_id not in img_sizes:
                continue
            w, h = img_sizes[image_id]
            gt_images.append(
                {"id": image_id, "width": w, "height": h, "file_name": f"{image_id}.png"}
            )
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, cx, cy, bw, bh = map(float, parts)
                        x1 = (cx - bw / 2) * w
                        y1 = (cy - bh / 2) * h
                        gt_anns.append(
                            {
                                "id": ann_id,
                                "image_id": image_id,
                                "category_id": 1,
                                "bbox": [x1, y1, bw * w, bh * h],
                                "area": bw * w * bh * h,
                                "iscrowd": 0,
                            }
                        )
                        ann_id += 1
        return {
            "images": gt_images,
            "annotations": gt_anns,
            "categories": [{"id": 1, "name": "RFI"}],
        }

    def preds_dict_to_coco_list(preds_by_image: dict) -> list:
        """Single-model xyxy + scores -> COCO dt list (same as ensemble run_wbf output shape)."""
        merged = []
        for image_id in image_ids:
            boxes_raw, scores_raw = preds_by_image.get(image_id, ([], []))
            for box, score in zip(boxes_raw, scores_raw):
                x1, y1, x2, y2 = box
                merged.append(
                    {
                        "image_id": int(image_id),
                        "category_id": cat_id,
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(score),
                    }
                )
        return merged

    def evaluate(preds: list) -> tuple[float, float]:
        if not preds:
            return 0.0, 0.0
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        gt = build_gt()
        for p in preds:
            p["category_id"] = 1
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(gt, f)
            gt_path = f.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(preds, f)
            dt_path = f.name
        coco_gt = COCO(gt_path)
        coco_dt = coco_gt.loadRes(dt_path)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        ev.summarize()
        sys.stdout = old_stdout
        os.unlink(gt_path)
        os.unlink(dt_path)
        return ev.stats[0], ev.stats[1]

    def run_and_cache(name: str, weights_path: str, imgsz: int) -> dict:
        tag = f"{name}_val"
        cache_path = cache_dir / f"{tag}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        from ultralytics import YOLO

        model = YOLO(weights_path)
        preds = {}
        for img_path in tqdm(img_paths, desc=f"  [{name}]"):
            image_id = int(img_path.stem)
            results = model.predict(
                source=str(img_path),
                imgsz=imgsz,
                conf=args.conf,
                iou=args.iou_nms,
                device=0,
                verbose=False,
                augment=args.tta,
            )
            r = results[0]
            boxes, scores = [], []
            if r.boxes is not None and len(r.boxes) > 0:
                for box, sc in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    boxes.append(box.tolist())
                    scores.append(float(sc))
            preds[image_id] = (boxes, scores)

        with open(cache_path, "wb") as f:
            pickle.dump(preds, f)
        return preds

    print("=" * 60)
    print("  Validation scores — solo (per model)")
    print("=" * 60)
    print(f"  Models: {len(available)}  |  cache: {cache_dir}")
    print()

    # rows: name, epoch_str, mAP50-95, mAP50, n_boxes
    rows: list[tuple[str, str, float, float, int]] = []
    for name, wpath, imgsz in available:
        raw = run_and_cache(name, wpath, imgsz)
        coco_list = preds_dict_to_coco_list(raw)
        m5095, m50 = evaluate(coco_list)
        nbox = sum(len(raw[i][0]) for i in raw)
        ep = epoch_by_name[name]
        rows.append((name, ep, m5095, m50, nbox))
        print(
            f"  {name:12s}  epoch={ep:>7}  mAP@50-95={m5095:.4f}  "
            f"mAP@50={m50:.4f}  boxes={nbox}"
        )

    ranked = sorted(rows, key=lambda r: r[2], reverse=True)
    print()
    print("  Summary table (sorted by mAP@50-95, descending)")
    hdr = (
        f"  {'#':>4}  {'model':<14}  {'epoch':>7}  "
        f"{'mAP@50-95':>10}  {'mAP@50':>8}  {'boxes':>8}"
    )
    print(hdr)
    print("  " + "-" * len(hdr.strip()))
    for i, (name, ep, m5095, m50, nbox) in enumerate(ranked, start=1):
        print(
            f"  {i:4d}  {name:<14}  {ep:>7}  {m5095:10.4f}  {m50:8.4f}  {nbox:8d}"
        )


if __name__ == "__main__":
    main()
