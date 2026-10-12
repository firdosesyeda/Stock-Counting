"""Split raw/ into coil photos vs. everything else using CLIP zero-shot classification.

Copies coil images to --out (default data/coils) and writes a CSV of scores so
borderline cases can be checked by eye.

    python scripts/sort_coils.py --src raw --out data/coils
"""
import argparse
import csv
import shutil
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

COIL = [
    "a photo of coiled pipe rolls tied with straps",
    "a photo of a coil of flexible hose",
    "a photo of stacked rolls of coiled cable",
]
OTHER = [
    "a photo of long straight pipes",
    "a photo of stacked pipe ends seen from the front",
    "a photo of a single corrugated pipe",
    "a product advertisement with text",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="raw")
    ap.add_argument("--out", default="data/coils")
    ap.add_argument("--thr", type=float, default=0.5, help="min summed coil probability")
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    args = ap.parse_args()

    model = CLIPModel.from_pretrained(args.model).eval()
    proc = CLIPProcessor.from_pretrained(args.model)
    prompts = COIL + OTHER
    files = sorted(Path(args.src).glob("*.jpg"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(0, len(files), 32):
        batch = files[i:i + 32]
        ims = [Image.open(f).convert("RGB") for f in batch]
        with torch.no_grad():
            inp = proc(text=prompts, images=ims, return_tensors="pt", padding=True)
            probs = model(**inp).logits_per_image.softmax(-1)
        for f, p in zip(batch, probs):
            score = float(p[: len(COIL)].sum())
            rows.append((f.name, round(score, 3)))
            if score >= args.thr:
                shutil.copy2(f, out / f.name)

    rows.sort(key=lambda r: -r[1])
    with open(Path(args.out).parent / "coil_scores.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([("file", "coil_score"), *rows])
    n = sum(s >= args.thr for _, s in rows)
    print(f"{n} / {len(rows)} images classified as coil -> {out}")


if __name__ == "__main__":
    main()
