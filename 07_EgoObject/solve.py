#!/usr/bin/env python3
"""EgoObjects 2D detection — YOLOv11n train, val, and Kaggle submission.

Competition: hkustgz-aiaa-4220-2025-fall-project-2
Metric: COCO mAP@0.50:0.95. Boxes are [x_min, y_min, width, height].
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    ROOT.parent / "data" / "ego-object" / "resource" / "data",
    ROOT.parent / "data" / "ego-object" / "data",
]
RUNS = ROOT / "runs"
WEIGHTS = RUNS / "yolo11n" / "weights" / "best.pt"


def find_data_dir() -> Path:
    for path in DATA_CANDIDATES:
        if (path / "dataset.yaml").exists():
            return path
    raise SystemExit("dataset.yaml not found. Unzip the competition archive under data/ego-object/.")


def prepare_yaml(data_dir: Path) -> Path:
    raw = yaml.safe_load((data_dir / "dataset.yaml").read_text())
    names = raw["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    cfg = {
        "path": str(data_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(names)},
    }
    out = ROOT / "dataset.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def train(data_yaml: Path) -> Path:
    RUNS.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolo11n.pt")
    t0 = time.time()
    results = model.train(
        data=str(data_yaml),
        epochs=30,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        pretrained=True,
        project=str(RUNS),
        name="yolo11n",
        exist_ok=True,
        patience=10,
        cos_lr=True,
        close_mosaic=5,
        seed=42,
        plots=False,
    )
    print(f"train_seconds={time.time() - t0:.1f}")
    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"parameters={n_params}")
    print(f"val_metrics={getattr(results, 'results_dict', {})}")
    if not WEIGHTS.exists():
        raise SystemExit(f"missing weights: {WEIGHTS}")
    return WEIGHTS


def image_id_map(data_dir: Path) -> dict[str, int]:
    meta = json.loads((data_dir / "test.json").read_text())
    mapping = {}
    for image in meta["images"]:
        stem = Path(image["file_name"]).stem
        mapping[stem] = int(image["id"])
    return mapping


def predict_and_submit(weights: Path, data_dir: Path) -> None:
    ids = image_id_map(data_dir)
    model = YOLO(str(weights))
    grouped: dict[int, list[dict]] = defaultdict(list)
    test_dir = data_dir / "images" / "test"
    for result in model.predict(source=str(test_dir), imgsz=640, conf=0.001, iou=0.6, device=0, stream=True, verbose=False):
        stem = Path(result.path).stem
        image_id = ids[stem]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        h, w = result.orig_shape
        for (x1, y1, x2, y2), category_id, score in zip(xyxy, cls, conf):
            x1 = float(max(0.0, min(x1, w - 1)))
            y1 = float(max(0.0, min(y1, h - 1)))
            bw = float(max(0.0, min(x2, w) - x1))
            bh = float(max(0.0, min(y2, h) - y1))
            if bw < 1 or bh < 1:
                continue
            grouped[image_id].append(
                {
                    "image_id": image_id,
                    "category_id": int(category_id),
                    "bbox": [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
                    "score": round(float(score), 5),
                }
            )

    rows = []
    for image_id in sorted(ids.values()):
        rows.append({"id": image_id, "predictions": json.dumps(grouped.get(image_id, []))})
    out = pd.DataFrame(rows)
    path = ROOT / "submission.csv"
    out.to_csv(path, index=False)
    n_det = sum(len(v) for v in grouped.values())
    print(f"wrote {path}  images={len(out)}  detections={n_det}")


def main() -> None:
    data_dir = find_data_dir()
    data_yaml = prepare_yaml(data_dir)
    print(f"data={data_dir}")
    if not WEIGHTS.exists():
        train(data_yaml)
    predict_and_submit(WEIGHTS, data_dir)


if __name__ == "__main__":
    main()
