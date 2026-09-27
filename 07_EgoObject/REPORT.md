# EgoObjects 2D Detection

Competition: [hkustgz-aiaa-4220-2025-fall-project-2](https://www.kaggle.com/competitions/hkustgz-aiaa-4220-2025-fall-project-2). Metric is COCO mAP@0.50:0.95.

## Method

YOLOv11n (Ultralytics), pretrained on COCO and fine-tuned on the provided YOLO labels. The 10 classes are pillow, box, book, bottle, chair, mug, door, shelf, plate, sofa, with `category_id` 0–9 matching `train.json`.

Training uses the official 10,000 / 1,000 split. Two validation labels with coordinates outside `[0, 1]` are skipped by Ultralytics (998 images, 4189 boxes). Test inference keeps boxes down to confidence 0.001 so the ranking metric can use the full score list.

## Hyperparameters

| Item | Value |
|---|---|
| Image size | 640 |
| Epochs | 30 (cosine LR, mosaic off for the last 5) |
| Batch size | 16 |
| Optimizer | AdamW, lr 7.14e-4 (Ultralytics auto) |
| GPU | 1× NVIDIA GeForce RTX 5060 Ti (16 GB) |
| Training time | about 16 minutes |
| Parameters | 2,591,790 |
| Weights | `runs/yolo11n/weights/best.pt` |

## Validation

Last epoch on the official validation split (`runs/yolo11n/results.csv`):

| Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|
| 0.943 | 0.913 | 0.957 | **0.806** |

The course Faster R-CNN / Deformable DETR baselines are listed at about 0.759–0.788 mAP@0.5:0.95.

## Reproduce

```bash
conda activate kaggle
python 07_EgoObject/solve.py
```

Data is expected at `data/ego-object/resource/data/` (competition zip already extracted). If `best.pt` exists, the script only reruns test inference and writes `07_EgoObject/submission.csv` (2000 rows, `id,predictions`).

Kaggle upload should include the student ID in the submission description.
