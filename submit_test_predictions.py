import argparse
import json
import pickle
import warnings
from pathlib import Path

from PIL import Image
from tqdm import tqdm

warnings.filterwarnings(
    "ignore",
    message=r"Zero area box skipped",
    category=UserWarning,
    module=r"ensemble_boxes\.ensemble_boxes_wbf",
)

REPO_ROOT = Path(__file__).resolve().parent


def _resolve(p: str | Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else REPO_ROOT / pp


parser = argparse.ArgumentParser(description="Generate test submission with WBF ensemble.")
parser.add_argument("--conf", type=float, default=0.001)
parser.add_argument(
    "--final-conf",
    type=float,
    default=0.80,
    help="Final confidence threshold applied after WBF.",
)
parser.add_argument("--iou-nms", type=float, default=0.5)
parser.add_argument("--iou-wbf", type=float, default=0.55)
parser.add_argument("--tta", action="store_true", default=True)
parser.add_argument("--cache-dir", type=str, default="pred_cache_v2")
parser.add_argument(
    "--anno-dir",
    type=str,
    default="clearsar_yolo/labels/test",
    help="Directory to write YOLO-format test annotations.",
)
parser.add_argument(
    "--5fold-only",
    action="store_true",
    dest="five_fold_only",
    help="Use fold0_p2..fold4_p2 only (equal weights).",
)
parser.add_argument("--out", type=str, default=None)
args = parser.parse_args()

ANN_FILE = REPO_ROOT / "ClearSAR/data/annotations/instances_train.json"
TEST_DIR = REPO_ROOT / "clearsar_yolo/images/test"
CACHE_DIR = REPO_ROOT / Path(args.cache_dir)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

with open(ANN_FILE) as f:
    coco = json.load(f)
cat_id = coco["categories"][0]["id"]

# Model registry
MODEL_CANDIDATES = [
    ("fold0_p2", "runs/detect/runs/clearsar/yolo11l_p2_fold0/weights/best.pt", 1280),
    ("fold1_p2", "runs/detect/runs/clearsar/yolo11l_p2_fold1/weights/best.pt", 1280),
    ("fold2_p2", "runs/detect/runs/clearsar/yolo11l_p2_fold2/weights/best.pt", 1280),
    ("fold3_p2", "runs/detect/runs/clearsar/yolo11l_p2_fold3/weights/best.pt", 1280),
    ("fold4_p2", "runs/detect/runs/clearsar/yolo11l_p2_fold4/weights/best.pt", 1280),
    ("exp4_p2", "runs/detect/runs/clearsar/yolo11l_p2_exp4/weights/best.pt", 1280),
    ("exp5_v8x", "runs/detect/runs/clearsar/yolov8x_exp5/weights/best.pt", 1280),
    ("exp22_v8l", "runs/detect/runs/clearsar/yolov8l_exp22/weights/best.pt", 800),
    ("exp1_v8m", "runs/detect/runs/clearsar/yolov8m_exp1/weights/best.pt", 640),
    ("yolov26L_p2_nwd", "Yolov11x_p2_nwd.pt", 1280),
]

FIVE_FOLD_NAMES = tuple(f"fold{i}_p2" for i in range(5))
if args.five_fold_only:
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

img_paths = sorted(TEST_DIR.glob("*.png"))
img_sizes = {}
for p in img_paths:
    img = Image.open(p)
    img_sizes[int(p.stem)] = (img.width, img.height)
    img.close()
image_ids = sorted(img_sizes.keys())


def run_and_cache(name: str, weights_path: str, imgsz: int):
    tag = f"{name}_test"
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


from ensemble_boxes import weighted_boxes_fusion


def run_wbf(selected_names, all_preds, weights=None):
    if weights is None:
        weights = [1.0] * len(selected_names)
    merged = []
    for image_id in image_ids:
        w, h = img_sizes[image_id]
        boxes_list, scores_list, labels_list = [], [], []
        for name in selected_names:
            boxes_raw, scores_raw = all_preds[name].get(image_id, ([], []))
            if boxes_raw:
                norm = [[b[0] / w, b[1] / h, b[2] / w, b[3] / h] for b in boxes_raw]
                norm = [[max(0, min(1, c)) for c in b] for b in norm]
            else:
                norm = []
            boxes_list.append(norm)
            scores_list.append(scores_raw)
            labels_list.append([0] * len(scores_raw))

        if all(len(b) == 0 for b in boxes_list):
            continue

        fused_boxes, fused_scores, _ = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=args.iou_wbf,
            skip_box_thr=0.0,
        )
        for box, score in zip(fused_boxes, fused_scores):
            x1, y1, x2, y2 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
            merged.append(
                {
                    "image_id": int(image_id),
                    "category_id": cat_id,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                }
            )
    return merged


print("=" * 60)
print("  ClearSAR Test Submission Ensemble")
print("=" * 60)
print(f"  Available models: {len(AVAILABLE)}")
for n, p, s in AVAILABLE:
    print(f"    {n}: {p} (imgsz={s})")
if args.five_fold_only:
    print("  Mode: 5-fold K only (WBF, equal weights on fold0_p2..fold4_p2)")
print(f"  TTA: {args.tta}")
print(f"  Final confidence (post-WBF): {args.final_conf:.2f}")

all_preds = {}
for name, wpath, imgsz in AVAILABLE:
    all_preds[name] = run_and_cache(name, wpath, imgsz)
    n_preds = sum(len(v[0]) for v in all_preds[name].values())
    print(f"    {name}: {n_preds} cached predictions")

model_names = [n for n, _, _ in AVAILABLE]
preds = run_wbf(model_names, all_preds)
preds = [p for p in preds if p["score"] >= args.final_conf]

if args.out:
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
elif args.five_fold_only:
    out_path = REPO_ROOT / "submission_5fold_test.json"
else:
    out_path = REPO_ROOT / "submission_exp2_test.json"

with open(out_path, "w") as f:
    json.dump(preds, f)

anno_dir = _resolve(args.anno_dir)
anno_dir.mkdir(parents=True, exist_ok=True)

preds_by_image = {}
for p in preds:
    preds_by_image.setdefault(int(p["image_id"]), []).append(p)

for image_id in image_ids:
    w, h = img_sizes[image_id]
    label_path = anno_dir / f"{image_id}.txt"
    lines = []
    for p in preds_by_image.get(image_id, []):
        x, y, bw, bh = p["bbox"]
        cx = (x + bw / 2.0) / w
        cy = (y + bh / 2.0) / h
        nw = bw / w
        nh = bh / h
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        nw = min(max(nw, 0.0), 1.0)
        nh = min(max(nh, 0.0), 1.0)
        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    with open(label_path, "w") as f:
        f.write("\n".join(lines))

print(f"\n  Predictions: {len(preds)}")
print(f"  Saved -> {out_path}")
print(f"  Test annotations -> {anno_dir}")
