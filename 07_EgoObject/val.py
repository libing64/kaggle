#!/usr/bin/env python3
"""Run the trained detector on each test image and save the result.

Each image is predicted on its own. Outputs:
  test_results/images/<name>.jpg   plotted boxes
  test_results/json/<stem>.json    detections for that image
"""

from __future__ import annotations

import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    ROOT.parent / "data" / "ego-object" / "resource" / "data" / "images" / "test",
    ROOT.parent / "data" / "ego-object" / "data" / "images" / "test",
]
WEIGHTS = ROOT / "runs" / "yolo11n" / "weights" / "best.pt"
OUT_DIR = ROOT / "test_results"


def find_test_dir() -> Path:
    for path in DATA_CANDIDATES:
        if path.is_dir() and any(path.glob("*.jpg")):
            return path
    raise SystemExit("test images not found under data/ego-object/")


def detections(result, names: dict) -> list[dict]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    height, width = result.orig_shape
    rows = []
    for (x1, y1, x2, y2), category_id, score in zip(xyxy, cls, conf):
        x1 = float(max(0.0, min(x1, width - 1)))
        y1 = float(max(0.0, min(y1, height - 1)))
        bw = float(max(0.0, min(x2, width) - x1))
        bh = float(max(0.0, min(y2, height) - y1))
        if bw < 1 or bh < 1:
            continue
        rows.append(
            {
                "category_id": int(category_id),
                "category": names[int(category_id)],
                "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x1 + bw, 2), round(y1 + bh, 2)],
                "bbox": [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
                "score": round(float(score), 5),
            }
        )
    return rows


def main() -> None:
    if not WEIGHTS.exists():
        raise SystemExit(f"missing weights: {WEIGHTS}")
    test_dir = find_test_dir()
    image_dir = OUT_DIR / "images"
    json_dir = OUT_DIR / "json"
    image_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(WEIGHTS))
    names = model.names
    images = sorted(test_dir.glob("*.jpg"))
    print(f"test images: {len(images)}")

    for index, path in enumerate(images, start=1):
        result = model.predict(
            source=str(path),
            imgsz=640,
            conf=0.25,
            iou=0.6,
            device=0,
            verbose=False,
        )[0]
        result.save(filename=str(image_dir / path.name))
        payload = {
            "file_name": path.name,
            "image_id": int(path.stem),
            "detections": detections(result, names),
        }
        (json_dir / f"{path.stem}.json").write_text(json.dumps(payload))
        if index == 1 or index % 100 == 0 or index == len(images):
            print(f"{index}/{len(images)}  {path.name}  boxes={len(payload['detections'])}")

    print(f"wrote {image_dir}")
    print(f"wrote {json_dir}")


if __name__ == "__main__":
    main()
