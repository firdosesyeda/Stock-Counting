"""Count coils in an image (or folder) and convert to total metres.

    python scripts/count_stock.py path/to/photo.jpg --metres-per-coil 500

Writes an annotated copy of each image to runs/count/ and prints a summary.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="image file or folder")
    ap.add_argument("--weights", default=str(ROOT / "runs/coil_v0/weights/best.pt"))
    ap.add_argument("--metres-per-coil", type=float, default=500)
    ap.add_argument("--conf", type=float, default=0.35)
    # Must match the training size: at 960 the 640-trained model misses whole coils.
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    if not Path(args.weights).exists():
        trained = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "runs").glob("*/weights/best.pt"))
        raise SystemExit(
            f"Model not found: {args.weights}\n"
            f"Trained models available: {', '.join(trained) or 'none - run scripts/train.py first'}"
        )
    if not Path(args.source).exists():
        raise SystemExit(f"Image or folder not found: {args.source}\nUse the real path to your photo(s).")

    model = YOLO(args.weights)
    out = ROOT / "runs/count"
    out.mkdir(parents=True, exist_ok=True)

    total_coils = 0
    for r in model.predict(args.source, conf=args.conf, imgsz=args.imgsz, max_det=500, verbose=False):
        counts = {name: 0 for name in model.names.values()}
        for c in r.boxes.cls.tolist():
            counts[model.names[int(c)]] += 1
        coils = counts.get("coil", 0)
        total_coils += coils
        r.save(filename=str(out / Path(r.path).name), labels=False)
        word = "coil" if coils == 1 else "coils"
        print(f"{Path(r.path).name}: {coils} {word} x {args.metres_per_coil:g} m = {coils * args.metres_per_coil:,.0f} m")

    print(f"\nTOTAL: {total_coils} coils = {total_coils * args.metres_per_coil:,.0f} metres")
    print(f"Annotated images saved to {out}")


if __name__ == "__main__":
    main()
