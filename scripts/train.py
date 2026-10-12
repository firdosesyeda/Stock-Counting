"""Fine-tune YOLO26n on the coil dataset.

CPU-only machine, so defaults are sized for that (nano model, 640px, batch 8).
On a CUDA GPU pass --device 0 and raise --batch.

    python scripts/train.py --epochs 60
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

# Absolute, so Ultralytics doesn't nest it under its global runs_dir (~/runs/detect/...).
RUNS = Path(__file__).resolve().parent.parent / "runs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/data.yaml")
    ap.add_argument("--model", default="weights/yolo26n.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--name", default="coil_v0")
    args = ap.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(f"Dataset not found: {args.data}\nRun scripts/build_dataset.py first.")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=4,
        patience=20,
        # Coil stacks are dense; allow many boxes per image.
        max_det=500,
        # Coils look the same mirrored; vertical flips don't occur on CCTV.
        fliplr=0.5,
        flipud=0.0,
        project=str(RUNS),
        name=args.name,
        exist_ok=True,
        plots=True,
    )
    metrics = model.val(data=args.data, device=args.device, max_det=500,
                        project=str(RUNS), name=f"{args.name}_val", exist_ok=True)
    print(f"mAP50={metrics.box.map50:.3f}  mAP50-95={metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
