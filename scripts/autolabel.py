"""Auto-label images with Grounding DINO (zero-shot) -> YOLO-format .txt labels.

These are *draft* labels: review/correct them in a labelling tool (CVAT, Label
Studio, Roboflow) before trusting the trained model. Re-runs skip images that
already have a label file, so it can be stopped and resumed.

    python scripts/autolabel.py --src data/coils --out labels/auto
"""
import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.ops import nms
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

PROMPT = "coil."


def drop_group_boxes(boxes):
    """Indices of boxes to keep, dropping any box that encloses the centres of 2+ others.

    Grounding DINO often adds one box around a whole stack on top of the per-coil boxes.
    """
    centres = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
    keep = []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        inside = sum(1 for j, (cx, cy) in enumerate(centres) if j != i and x1 < cx < x2 and y1 < cy < y2)
        if inside < 2:
            keep.append(i)
    return keep


def detect(proc, model, im, box_thr, text_thr):
    """Boxes (N,4 xyxy) and scores (N,) for PROMPT in one PIL image."""
    inp = proc(images=im, text=PROMPT, return_tensors="pt")
    with torch.no_grad():
        res = model(**inp)
    r = proc.post_process_grounded_object_detection(
        res, inp.input_ids, threshold=box_thr, text_threshold=text_thr, target_sizes=[im.size[::-1]]
    )[0]
    return r["boxes"], r["scores"]


def detect_tiled(proc, model, im, box_thr, text_thr, overlap=0.2):
    """Full image plus 2x2 overlapping tiles, so each coil in a dense stack is larger to the model."""
    w, h = im.size
    tw, th = int(w * (0.5 + overlap / 2)), int(h * (0.5 + overlap / 2))
    b, s = detect(proc, model, im, box_thr, text_thr)
    boxes, scores = [b], [s]
    for x0 in (0, w - tw):
        for y0 in (0, h - th):
            b, s = detect(proc, model, im.crop((x0, y0, x0 + tw, y0 + th)), box_thr, text_thr)
            if not len(b):
                continue
            b = b + torch.tensor([x0, y0, x0, y0], dtype=b.dtype)
            # Drop boxes touching a tile edge that is inside the image: the coil is cut off there,
            # and the full-image pass or the neighbouring tile sees it whole.
            m = 0.02 * max(w, h)
            cut = torch.zeros(len(b), dtype=torch.bool)
            if x0 > 0: cut |= b[:, 0] < x0 + m
            if x0 + tw < w: cut |= b[:, 2] > x0 + tw - m
            if y0 > 0: cut |= b[:, 1] < y0 + m
            if y0 + th < h: cut |= b[:, 3] > y0 + th - m
            boxes.append(b[~cut]); scores.append(s[~cut])
    return torch.cat(boxes), torch.cat(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/coils")
    ap.add_argument("--out", default="labels/auto")
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--box-thr", type=float, default=0.25)
    ap.add_argument("--text-thr", type=float, default=0.2)
    ap.add_argument("--preview", default="labels/auto_preview")
    ap.add_argument("--tiles", action="store_true", help="also detect on 2x2 tiles (5x slower, better on stacks)")
    ap.add_argument("--only", help="text file of image names to process (one per line)")
    args = ap.parse_args()

    torch.set_num_threads(torch.get_num_threads() or 8)
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).eval()

    out, preview = Path(args.out), Path(args.preview)
    out.mkdir(parents=True, exist_ok=True)
    preview.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.src).glob("*.jpg"))
    if args.only:
        wanted = set(Path(args.only).read_text().split())
        files = [f for f in files if f.name in wanted]

    for i, f in enumerate(files, 1):
        dst = out / f"{f.stem}.txt"
        if dst.exists():
            continue
        im = Image.open(f).convert("RGB")
        w, h = im.size
        run = detect_tiled if args.tiles else detect
        boxes, scores = run(proc, model, im, args.box_thr, args.text_thr)

        lines = []
        if len(boxes):
            B = boxes[nms(boxes, scores, 0.5)]
            for k in drop_group_boxes(B.tolist()):
                x1, y1, x2, y2 = B[k].tolist()
                lines.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
                ImageDraw.Draw(im).rectangle((x1, y1, x2, y2), outline="lime", width=max(2, w // 250))
        dst.write_text("\n".join(lines))
        im.thumbnail((640, 640)); im.save(preview / f.name, quality=85)
        print(f"[{i}/{len(files)}] {f.name}: {len(lines)} boxes", flush=True)


if __name__ == "__main__":
    main()
