# Coil Stock Counting Project Report

**Prepared:** 10 October 2026  
**Purpose:** Detect visible coils in stock photos and estimate total stock length from the detected coil count.

## Project overview

This project uses object detection to locate individual coils in photos. It currently has one detection class, `coil`. The counting script reports the number of detections and multiplies that number by a configurable metres-per-coil value (500 m by default).

The workflow is:

1. `scripts/sort_coils.py` uses CLIP (`openai/clip-vit-base-patch32`) to sort likely coil photos from other images and records scores in `data/coil_scores.csv`.
2. Images are labelled in YOLO format. `scripts/autolabel.py` can create draft labels with Grounding DINO (`IDEA-Research/grounding-dino-base`); labels should be reviewed and corrected.
3. `scripts/build_dataset.py` creates train/validation datasets from the images and labels.
4. `scripts/train.py` fine-tunes the YOLO26n model using the dataset.
5. `scripts/count_stock.py` predicts and counts coils in a supplied image or folder.

## How the data is used

- `raw/` is the starting image collection. The CLIP sorting script scores each image against coil and non-coil descriptions, copies images above its threshold (0.5 by default) to `data/coils/`, and writes scores to `data/coil_scores.csv` for review. CLIP helps select candidate photos; it does not count the coils.
- YOLO label files contain one bounding box per visible coil, in normalized YOLO format. The Grounding DINO script can draft these boxes and preview images, but its own documentation calls them drafts. `labels/fixed/` stores hand-corrected labels. Empty labels are excluded when building the dataset so missed auto-labels do not train the detector as if a real coil photo contained no coils.
- The dataset builder pairs each `.jpg` with its same-name `.txt`, copies pairs into `images/train`, `images/val`, `labels/train`, and `labels/val`, and writes `data.yaml`. That YAML tells YOLO where the images and labels are and maps class 0 to `coil`.
- The saved datasets contain 99 train and 17 validation images in v0, and 100 train and 17 validation images in v1. The v0 and v1 validation image filenames are the same, so validation results can be compared on the same photos. The v1 dataset also has updated validation annotations, so the reported scores compare each model against the v1 labels.
- `data/review.csv` is a separate human review table with estimated visible coil counts and label-quality notes. The comparison script can use these estimates for count-error analysis; they are distinct from the bounding-box metrics reported below.

The model learns from training images and their coil boxes. Validation images are held out from weight updates and are used to measure detection quality during training and evaluation.

## Models and training runs

Both detector runs use the pretrained `weights/yolo26n.pt` checkpoint, Ultralytics YOLO26n, one class (`coil`), 640-pixel image size, batch size 8, CPU, and a configured maximum of 100 epochs. YOLO26n is the nano/small-footprint variant: it was chosen as a practical starting detector for this modest, single-class dataset and CPU-only training environment, where a larger model would require more compute and time. This is a project configuration choice, not evidence that YOLO26n is universally the best architecture. The saved run configuration files identify the datasets used.

| Run | Dataset | Train images | Validation images | Best checkpoint |
|---|---|---:|---:|---|
| `coil_v0` | `dataset` | 99 | 17 | `runs/coil_v0/weights/best.pt` |
| `coil_v1` | `dataset_v1` | 100 | 17 | `runs/coil_v1/weights/best.pt` |

The project also includes CLIP and Grounding DINO for image sorting and draft labelling. They are preprocessing tools, not the final coil-counting detector.

## Validation metrics

The saved `runs/model_metrics.txt` records validation metrics for the two checkpoints. These are detection metrics on the configured validation dataset, not measurements on the separate `Test.jpg` photo.

| Run | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 |
|---|---:|---:|---:|---:|
| `coil_v0` | 0.7661 | 0.3948 | 0.4533 | 0.3350 |
| `coil_v1` | 0.7098 | 0.7108 | 0.7012 | 0.5062 |

The metrics were measured on the same 17 validation photos using the v1 validation labels. On this comparison, v1 recall rose from 0.3948 to 0.7108, mAP@0.50 from 0.4533 to 0.7012, and mAP@0.50:0.95 from 0.3350 to 0.5062. Precision fell from 0.7661 to 0.7098, so v1 found more labelled coils overall at the cost of a somewhat higher rate of false detections. This is evidence of improvement in recall and box accuracy on this small held-out set, not a guarantee of the same improvement on new photos.

## What changed from v0 to v1

The model family and core training settings stayed the same. The saved run configurations show the same YOLO26n pretrained weights, 640 image size, batch 8, CPU, 100-epoch maximum, and augmentation settings. The main recorded change is the training data: v1 uses `dataset_v1` rather than `dataset`.

Comparing the saved datasets shows that v1 contains one additional training image and 27 changed training label files relative to v0. The workspace has 32 files under `labels/fixed`; 27 of those corrected labels appear in the v1 training split. The validation images remain the same 17 photos, and five of their labels changed in v1. This points to label correction/refinement as the primary intended change, plus one added training example. The available artifacts do not identify the exact edit made to each box or provide a change log, so more specific claims about individual annotation edits would be guesswork.

The evaluation supports a meaningful improvement in recall and mAP on the shared validation images, while precision decreased. The test-photo outputs also changed substantially, from 4 detections with v0 to 15 with v1. Since the actual count in `Test.jpg` has not been independently confirmed, that difference alone cannot establish which prediction is correct.

## Test image output

The user ran both models on `C:\Users\Firdose\Documents\Souveno\Test.jpg` with the counting script defaults: confidence threshold 0.35, image size 640, and 500 metres per coil.

| Run | Detected coils | Estimated length |
|---|---:|---:|
| `coil_v0` | 4 | 2,000 m |
| `coil_v1` | 15 | 7,500 m |

These are the observed model outputs, not a verified ground-truth count. The large difference means the image should be reviewed visually and the true coil count confirmed before using either estimate as inventory. The script saves annotated output to `runs/count/Test.jpg`; running the second model overwrites the first output at that path unless it is copied or renamed between runs. The separate validation plots and weights are in each run folder; v1 also has an epoch history in `runs/coil_v1/results.csv`.

## Useful files

- `scripts/count_stock.py` — run inference and print coil and metre totals.
- `scripts/train.py` — training entry point.
- `scripts/compare_models.py` — compare detections against reviewed estimates in `data/review.csv` on the held-out validation images.
- `data/review.csv` — manual estimated counts and label review status.
- `runs/coil_v0/` and `runs/coil_v1/` — training plots, validation visualizations, configs, and checkpoints.
- `runs/model_metrics.txt` — saved validation metric summary.

## Reproduction commands

From the project folder, run either checkpoint on the test photo:

```powershell
python scripts/count_stock.py "C:\Users\Firdose\Documents\Souveno\Test.jpg" --weights "runs\coil_v0\weights\best.pt"
python scripts/count_stock.py "C:\Users\Firdose\Documents\Souveno\Test.jpg" --weights "runs\coil_v1\weights\best.pt"
```

For a reviewed, image-by-image comparison on the validation set:

```powershell
python scripts/compare_models.py "runs\coil_v0\weights\best.pt" "runs\coil_v1\weights\best.pt"
```
