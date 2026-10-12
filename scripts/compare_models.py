"""Compare trained models on the same held-out photos: counting error and box accuracy.

    python scripts/compare_models.py runs/coil_v0/weights/best.pt runs/coil_v1/weights/best.pt

Counting error uses the hand-estimated visible coil counts in data/review.csv
(est_coils), so it measures what the client cares about: "how many coils are there".
"""
import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="+")
    ap.add_argument("--data", default=str(ROOT / "dataset_v1/data.yaml"))
    ap.add_argument("--val-images", default=str(ROOT / "dataset_v1/images/val"))
    ap.add_argument("--review", default=str(ROOT / "data/review.csv"))
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    truth = {r["file"]: (int(r["est_coils"]), r["type"]) for r in csv.DictReader(open(args.review))}
    results = {}
    for w in args.weights:
        model = YOLO(w)
        preds = {Path(r.path).name: len(r.boxes)
                 for r in model.predict(args.val_images, conf=args.conf, imgsz=args.imgsz, max_det=500, verbose=False)}
        m = model.val(data=args.data, imgsz=args.imgsz, max_det=500, verbose=False, plots=False,
                      project=str(ROOT / "runs"), name="_compare", exist_ok=True)
        results[w] = (preds, m.box.map50)

    names = sorted(next(iter(results.values()))[0])
    labels = [Path(w).parent.parent.name for w in args.weights]
    print(f"\n{'image':24} {'type':6} {'actual':>6} " + " ".join(f"{l:>9}" for l in labels))
    for n in names:
        est, typ = truth[n]
        print(f"{n:24} {typ:6} {est:>6} " + " ".join(f"{results[w][0][n]:>9}" for w in args.weights))

    print()
    for w, label in zip(args.weights, labels):
        preds, map50 = results[w]
        total_est = sum(truth[n][0] for n in names)
        total_pred = sum(preds[n] for n in names)
        err = sum(abs(preds[n] - truth[n][0]) for n in names)
        stack = [n for n in names if truth[n][1] == "stack"]
        stack_pct = 100 * sum(preds[n] for n in stack) / max(1, sum(truth[n][0] for n in stack))
        print(f"{label}: counted {total_pred}/{total_est} coils ({100 * total_pred / total_est:.0f}%), "
              f"total abs error {err}, stacks found {stack_pct:.0f}% of coils, mAP50 {map50:.3f}")


if __name__ == "__main__":
    main()
