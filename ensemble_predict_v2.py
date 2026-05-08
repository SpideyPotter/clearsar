import json
import pickle
import argparse
import sys
import os
import tempfile
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from itertools import combinations
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Zero area box skipped",
    category=UserWarning,
    module=r"ensemble_boxes\.ensemble_boxes_wbf",
)

REPO_ROOT = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["val", "test"], default="val")
parser.add_argument("--conf", type=float, default=0.001)
parser.add_argument("--iou-nms", type=float, default=0.5)
parser.add_argument("--iou-wbf", type=float, default=0.55)
parser.add_argument("--tta", action="store_true", default=True)
parser.add_argument("--cache-dir", type=str, default="pred_cache_v2")
parser.add_argument("--ablation", action="store_true", help="Greedy forward selection")
parser.add_argument(
    "--5fold-only",
    action="store_true",
    dest="five_fold_only",
    help="WBF ensemble of fold0_p2..fold4_p2 only (equal weights). "
    "Default output: submission_5fold_{val|test}.json under repo root.",
)
parser.add_argument("--out", type=str, default=None)
args = parser.parse_args()


def _resolve(p: str | Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else REPO_ROOT / pp


ANN_FILE = REPO_ROOT / "ClearSAR/data/annotations/instances_train.json"
VAL_DIR = REPO_ROOT / "clearsar_yolo/images/val"
TEST_DIR = REPO_ROOT / "clearsar_yolo/images/test"
VAL_LABELS = REPO_ROOT / "clearsar_yolo/labels/val"
CACHE_DIR = REPO_ROOT / Path(args.cache_dir)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

with open(ANN_FILE) as f:
    coco = json.load(f)
cat_id = coco["categories"][0]["id"]

# ── Model registry ────────────────────────────────────────────────────────────
MODEL_CANDIDATES = [
    # K-fold YOLO11l-P2
     ("fold0_p2",  "runs/detect/runs/clearsar/yolo11l_p2_fold0/weights/best.pt", 1280),
    ("fold1_p2",  "runs/detect/runs/clearsar/yolo11l_p2_fold1/weights/best.pt", 1280),
    ("fold2_p2",  "runs/detect/runs/clearsar/yolo11l_p2_fold2/weights/best.pt", 1280),
     ("fold3_p2",  "runs/detect/runs/clearsar/yolo11l_p2_fold3/weights/best.pt", 1280),
     ("fold4_p2",  "runs/detect/runs/clearsar/yolo11l_p2_fold4/weights/best.pt", 1280),
    # New architectures (Ultralytics nests under runs/detect/ when project=runs/clearsar)
    # RT-DETR-x: exported best epoch (see EXPERIMENT_2.md); run dir may be absent locally
    # ("rtdetr_x",  "RT_detr_56.pt", 1280),
    #  ("yolo11x",   "runs/detect/runs/clearsar/yolo11x_exp8/weights/best.pt",     1280),
    #  ("yolov9e",   "runs/detect/runs/clearsar/yolov9e_exp9/weights/best.pt",     1280),
    # # Experiment 1 survivors (drop exp2 which was undertrained)
     ("exp4_p2",   "runs/detect/runs/clearsar/yolo11l_p2_exp4/weights/best.pt",  1280),
     ("exp5_v8x",  "runs/detect/runs/clearsar/yolov8x_exp5/weights/best.pt",     1280),
    #  ("exp3_y11m", "runs/detect/runs/clearsar/yolo11m_exp3/weights/best.pt",      1280),
     ("exp22_v8l", "runs/detect/runs/clearsar/yolov8l_exp22/weights/best.pt",     800),
     ("exp1_v8m",  "runs/detect/runs/clearsar/yolov8m_exp1/weights/best.pt",      640),
    # ("exp_14.py","runs/clearsar_modified/weights/best.pt",1280),
    # ("Yolov26Lp2", "Yolov26Lp2.pt", 1280),
    ("yolov26L_p2_nwd", "Yolov11x_p2_nwd.pt", 1280),

]

FIVE_FOLD_NAMES = tuple(f"fold{i}_p2" for i in range(5))
if args.five_fold_only:
    if args.ablation:
        print("  [--5fold-only] ignoring --ablation (fixed 5-model pool).")
        args.ablation = False
    candidates = [t for t in MODEL_CANDIDATES if t[0] in FIVE_FOLD_NAMES]
else:
    candidates = MODEL_CANDIDATES

AVAILABLE = [
    (n, str(_resolve(p)), s)
    for n, p, s in candidates
    if _resolve(p).is_file()
]

if args.five_fold_only and len(AVAILABLE) < 5:
    missing = set(FIVE_FOLD_NAMES) - {n for n, _, _ in AVAILABLE}
    raise SystemExit(
        f"--5fold-only requires all 5 fold checkpoints. Found {len(AVAILABLE)}/5. Missing: {sorted(missing)}"
    )

img_dir = VAL_DIR if args.split == "val" else TEST_DIR
img_paths = sorted(img_dir.glob("*.png"))
img_sizes = {}
for p in img_paths:
    img = Image.open(p)
    img_sizes[int(p.stem)] = (img.width, img.height)
    img.close()
image_ids = sorted(img_sizes.keys())

# ── GT builder ────────────────────────────────────────────────────────────────
def build_gt():
    gt_images, gt_anns = [], []
    ann_id = 1
    for lbl_path in sorted(VAL_LABELS.glob("*.txt")):
        image_id = int(lbl_path.stem)
        if image_id not in img_sizes:
            continue
        w, h = img_sizes[image_id]
        gt_images.append({"id": image_id, "width": w, "height": h,
                          "file_name": f"{image_id}.png"})
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, cx, cy, bw, bh = map(float, parts)
                    x1 = (cx - bw / 2) * w
                    y1 = (cy - bh / 2) * h
                    gt_anns.append({
                        "id": ann_id, "image_id": image_id,
                        "category_id": 1, "bbox": [x1, y1, bw * w, bh * h],
                        "area": bw * w * bh * h, "iscrowd": 0,
                    })
                    ann_id += 1
    return {"images": gt_images, "annotations": gt_anns,
            "categories": [{"id": 1, "name": "RFI"}]}

# ── Inference + caching ───────────────────────────────────────────────────────
def run_and_cache(name, weights_path, imgsz):
    tag = f"{name}_{args.split}"
    cache_path = CACHE_DIR / f"{tag}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    from ultralytics import YOLO
    model = YOLO(weights_path)
    preds = {}
    for img_path in tqdm(img_paths, desc=f"  [{name}]"):
        image_id = int(img_path.stem)
        results = model.predict(
            source=str(img_path), imgsz=imgsz, conf=args.conf,
            iou=args.iou_nms, device=0, verbose=False, augment=args.tta,
        )
        r = results[0]
        boxes, scores = [], []
        if r.boxes is not None and len(r.boxes) > 0:
            for box, sc in zip(r.boxes.xyxy.cpu().numpy(),
                               r.boxes.conf.cpu().numpy()):
                boxes.append(box.tolist())
                scores.append(float(sc))
        preds[image_id] = (boxes, scores)

    with open(cache_path, "wb") as f:
        pickle.dump(preds, f)
    return preds

# ── WBF ───────────────────────────────────────────────────────────────────────
from ensemble_boxes import weighted_boxes_fusion

def run_wbf(selected_names, all_preds, weights=None, iou_thr=None):
    iou_thr = iou_thr or args.iou_wbf
    if weights is None:
        weights = [1.0] * len(selected_names)
    merged = []
    for image_id in image_ids:
        w, h = img_sizes[image_id]
        boxes_list, scores_list, labels_list = [], [], []
        for name in selected_names:
            boxes_raw, scores_raw = all_preds[name].get(image_id, ([], []))
            if boxes_raw:
                norm = [[b[0]/w, b[1]/h, b[2]/w, b[3]/h] for b in boxes_raw]
                norm = [[max(0, min(1, c)) for c in b] for b in norm]
            else:
                norm = []
            boxes_list.append(norm)
            scores_list.append(scores_raw)
            labels_list.append([0] * len(scores_raw))

        if all(len(b) == 0 for b in boxes_list):
            continue

        fused_boxes, fused_scores, _ = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=weights, iou_thr=iou_thr, skip_box_thr=0.0,
        )
        for box, score in zip(fused_boxes, fused_scores):
            x1, y1, x2, y2 = box[0]*w, box[1]*h, box[2]*w, box[3]*h
            merged.append({
                "image_id": int(image_id), "category_id": cat_id,
                "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                "score": float(score),
            })
    return merged

# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(preds):
    if not preds:
        return 0.0, 0.0
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = build_gt()
    for p in preds:
        p["category_id"] = 1
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(gt, f); gt_path = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(preds, f); dt_path = f.name
    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(dt_path)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate(); ev.accumulate()
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    ev.summarize()
    sys.stdout = old_stdout
    os.unlink(gt_path); os.unlink(dt_path)
    return ev.stats[0], ev.stats[1]

# ── Main ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  ClearSAR Experiment 2 -- Ensemble v2")
print("=" * 60)
print(f"  Available models: {len(AVAILABLE)}")
for n, p, s in AVAILABLE:
    print(f"    {n}: {p} (imgsz={s})")
print(f"  Split: {args.split}, TTA: {args.tta}")
if args.five_fold_only:
    print("  Mode: 5-fold K only (WBF, equal weights on fold0_p2..fold4_p2)")

all_preds = {}
for name, wpath, imgsz in AVAILABLE:
    all_preds[name] = run_and_cache(name, wpath, imgsz)
    n = sum(len(v[0]) for v in all_preds[name].values())
    print(f"    {name}: {n} cached predictions")

model_names = [n for n, _, _ in AVAILABLE]

if args.ablation and args.split == "val":
    print("\n" + "=" * 60)
    print("  Greedy Forward Selection (ablation)")
    print("=" * 60)

    # Evaluate each model solo first
    solo_scores = {}
    for name in model_names:
        preds = run_wbf([name], all_preds, weights=[1.0])
        m5095, m50 = evaluate(preds)
        solo_scores[name] = m5095
        print(f"    Solo {name}: mAP50-95={m5095:.4f}  mAP50={m50:.4f}")

    ranked = sorted(solo_scores.keys(), key=lambda n: solo_scores[n], reverse=True)
    print(f"\n  Ranked: {ranked}")

    selected = [ranked[0]]
    best_score = solo_scores[ranked[0]]
    print(f"\n  Start: [{selected[0]}] mAP50-95={best_score:.4f}")

    for rnd in range(1, len(ranked)):
        best_add, best_new_score = None, best_score
        for candidate in ranked:
            if candidate in selected:
                continue
            trial = selected + [candidate]
            preds = run_wbf(trial, all_preds)
            m5095, m50 = evaluate(preds)
            delta = m5095 - best_score
            print(f"    + {candidate:>12s} -> mAP50-95={m5095:.4f} (delta={delta:+.4f})")
            if m5095 > best_new_score:
                best_new_score = m5095
                best_add = candidate

        if best_add and best_new_score > best_score:
            selected.append(best_add)
            best_score = best_new_score
            print(f"  >> Added {best_add}. Selected={selected}  mAP50-95={best_score:.4f}")
        else:
            print(f"  >> No improvement. Stopping.")
            break

    print(f"\n  OPTIMAL SUBSET ({len(selected)} models):")
    for s in selected:
        print(f"    {s}")
    print(f"  mAP@50-95: {best_score:.4f}")

    preds = run_wbf(selected, all_preds)
    m5095, m50 = evaluate(preds)
    print(f"\n  Final mAP@50-95 : {m5095:.4f}")
    print(f"  Final mAP@50    : {m50:.4f}")

    model_names = selected  # use for output

else:
    preds = run_wbf(model_names, all_preds)

    if args.split == "val":
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        gt = build_gt()
        for p in preds:
            p["category_id"] = 1
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(gt, f); gt_path = f.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(preds, f); dt_path = f.name
        coco_gt = COCO(gt_path)
        coco_dt = coco_gt.loadRes(dt_path)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate(); ev.accumulate(); ev.summarize()
        os.unlink(gt_path); os.unlink(dt_path)
        print(f"\n  mAP@50-95 : {ev.stats[0]:.4f}")
        print(f"  mAP@50    : {ev.stats[1]:.4f}")

if args.out:
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
elif args.five_fold_only:
    out_path = REPO_ROOT / f"submission_5fold_{args.split}.json"
else:
    out_path = REPO_ROOT / f"submission_exp2_{args.split}.json"

with open(out_path, "w") as f:
    json.dump(preds, f)
print(f"\n  Predictions: {len(preds)}")
print(f"  Saved -> {out_path}")
