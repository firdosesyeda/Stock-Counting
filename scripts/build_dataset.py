"""Split labelled images into a YOLO dataset (train/val) and write data.yaml.

Expects YOLO-format label files (one .txt per image, same stem) in --labels.
Images without a label file, or with an empty one, are skipped: in this data an
empty label almost always means the auto-labeller missed the objects, and
training on it would teach the model to ignore real coils.

    python scripts/build_dataset.py --images data/coils --labels labels/auto --out dataset

--override DIR replaces individual label files with hand-corrected ones (e.g. labels/fixed).
--same-split-as DIR keeps an existing dataset's val images in val, so two models can be
compared on the same held-out photos.
"""
import argparse
import random
import shutil
from pathlib import Path

CLASSES = ["coil"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/coils")
    ap.add_argument("--labels", default="labels/auto")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--override", help="folder of corrected labels that take priority over --labels")
    ap.add_argument("--same-split-as", help="existing dataset folder whose val split to reuse")
    args = ap.parse_args()

    images, labels, out = Path(args.images), Path(args.labels), Path(args.out)
    override = Path(args.override) if args.override else None
    pairs = []
    for img in sorted(images.glob("*.jpg")):
        lbl = labels / f"{img.stem}.txt"
        if override and (override / f"{img.stem}.txt").exists():
            lbl = override / f"{img.stem}.txt"
        if lbl.exists() and lbl.read_text().strip():
            pairs.append((img, lbl))
    if not pairs:
        raise SystemExit(
            f"No labelled images found (labels in {labels}, images in {images}).\n"
            "Label the photos in Roboflow first, export as YOLOv8, unzip the export and copy its\n"
            "images/ and labels/ files into those two folders, then run this again."
        )

    if args.same_split_as:
        val_names = {p.name for p in (Path(args.same_split_as) / "images" / "val").glob("*.jpg")}
        splits = {"val": [p for p in pairs if p[0].name in val_names],
                  "train": [p for p in pairs if p[0].name not in val_names]}
    else:
        random.Random(args.seed).shuffle(pairs)
        n_val = max(1, int(len(pairs) * args.val))
        splits = {"val": pairs[:n_val], "train": pairs[n_val:]}

    if out.exists():
        shutil.rmtree(out)
    for split, items in splits.items():
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        for img, lbl in items:
            shutil.copy2(img, out / "images" / split / img.name)
            shutil.copy2(lbl, out / "labels" / split / lbl.name)

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n"
    )
    print(f"train={len(splits['train'])} val={len(splits['val'])} -> {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
