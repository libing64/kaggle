#!/usr/bin/env python3
"""RSNA knee abnormality detection.

Twelve study-level labels. Only 58 training studies are labeled, and the hidden
test set has no radiology report, so the submission model reads MRI slices.

Uses one fluid-sensitive sagittal series and one coronal series per study.
Three central slices are stacked as a 3-channel image for a pretrained ResNet-18.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "rsna-knee"
LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_series(group: pd.DataFrame) -> list[str]:
    chosen = []
    for plane in ("Sagittal", "Coronal"):
        subset = group[group["Anatomical_Plane"] == plane]
        fluid = subset[subset["Fluid_Sensitive"] == 1]
        pool = fluid if len(fluid) else subset
        if len(pool):
            chosen.append(str(pool.iloc[0]["SeriesInstanceUID"]))
    if not chosen and len(group):
        chosen.append(str(group.iloc[0]["SeriesInstanceUID"]))
    return chosen


def series_to_tensor(series_dir: Path) -> torch.Tensor | None:
    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        return None
    slices = []
    for path in files:
        try:
            ds = pydicom.dcmread(path)
            image = ds.pixel_array.astype(np.float32)
        except Exception:
            continue
        image = image * float(getattr(ds, "RescaleSlope", 1) or 1) + float(getattr(ds, "RescaleIntercept", 0) or 0)
        slices.append((int(getattr(ds, "InstanceNumber", len(slices))), image))
    if not slices:
        return None
    slices.sort(key=lambda item: item[0])
    center = len(slices) // 2
    planes = []
    for index in (max(0, center - 2), center, min(len(slices) - 1, center + 2)):
        image = slices[index][1]
        low, high = np.percentile(image, [1, 99])
        image = np.clip((image - low) / max(high - low, 1e-3), 0, 1)
        tensor = torch.from_numpy(image)[None, None]
        tensor = torch.nn.functional.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
        planes.append(tensor[0, 0])
    return torch.stack(planes, 0).float()


class StudyDataset(Dataset):
    def __init__(self, frames: list[torch.Tensor], targets: np.ndarray | None):
        self.frames = frames
        self.targets = targets

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int):
        image = self.frames[index]
        if self.targets is None:
            return image
        return image, torch.tensor(self.targets[index], dtype=torch.float32)


def build_model() -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    return model.to(DEVICE)


def train_one(model: nn.Module, loader: DataLoader, epochs: int = 8) -> None:
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        for images, target in loader:
            images = images.to(DEVICE)
            target = target.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(images), target)
            loss.backward()
            opt.step()


@torch.no_grad()
def predict(model: nn.Module, frames: list[torch.Tensor]) -> np.ndarray:
    model.eval()
    if not frames:
        return np.full((1, len(LABELS)), 0.5)
    batch = torch.stack(frames).to(DEVICE)
    logits = []
    for start in range(0, len(batch), 16):
        logits.append(model(batch[start : start + 16]).sigmoid().cpu().numpy())
    return np.concatenate(logits, axis=0)


def study_frames(series: pd.DataFrame, root: Path) -> dict[str, list[torch.Tensor]]:
    frames: dict[str, list[torch.Tensor]] = {}
    for study_id, group in series.groupby("StudyInstanceUID"):
        views = []
        for series_id in pick_series(group):
            tensor = series_to_tensor(root / str(study_id) / series_id)
            if tensor is not None:
                views.append(tensor)
        if views:
            frames[str(study_id)] = views
    return frames


def macro_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    scores = []
    for col in range(y_true.shape[1]):
        if len(np.unique(y_true[:, col])) < 2:
            continue
        scores.append(roc_auc_score(y_true[:, col], y_prob[:, col]))
    return float(np.mean(scores)) if scores else float("nan")


def report_oof(labeled: pd.DataFrame) -> float:
    y = labeled[LABELS].to_numpy(dtype=np.float32)
    text = labeled["Report"].fillna("")
    oof = np.zeros_like(y)
    for tr, va in KFold(n_splits=5, shuffle=True, random_state=42).split(y):
        model = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=8000)),
                ("clf", OneVsRestClassifier(LogisticRegression(max_iter=400, C=1.0))),
            ]
        )
        model.fit(text.iloc[tr], y[tr])
        oof[va] = model.predict_proba(text.iloc[va])
    score = macro_auc(y, oof)
    print(f"report TF-IDF 5-fold macro-AUC {score:.4f}  n={len(labeled)}")
    return score


def main() -> None:
    print(f"device={DEVICE}")
    train = pd.read_csv(DATA / "train.csv")
    series = pd.read_csv(DATA / "train_series.csv")
    labeled = train.dropna(subset=LABELS).reset_index(drop=True)
    report_oof(labeled)
    prevalence = labeled[LABELS].mean().to_numpy(dtype=np.float32)
    labeled_ids = set(labeled["StudyInstanceUID"])
    train_frames = study_frames(series[series["StudyInstanceUID"].isin(labeled_ids)], DATA / "train_series")
    labeled = labeled[labeled["StudyInstanceUID"].isin(train_frames)].reset_index(drop=True)
    print(f"labeled studies with images: {len(labeled)}")
    image_model = None
    if len(labeled) >= 8:
        y = labeled[LABELS].to_numpy(dtype=np.float32)
        groups = labeled["StudyInstanceUID"].to_numpy()
        oof = np.zeros_like(y)
        cv = GroupKFold(n_splits=min(5, len(labeled)))
        for fold, (tr, va) in enumerate(cv.split(y, y[:, 0], groups), start=1):
            train_x, train_y = [], []
            for index in tr:
                study_id = groups[index]
                for view in train_frames[study_id]:
                    train_x.append(view)
                    train_y.append(y[index])
            model = build_model()
            loader = DataLoader(StudyDataset(train_x, np.stack(train_y)), batch_size=8, shuffle=True)
            train_one(model, loader)
            val_views = [train_frames[groups[index]] for index in va]
            probs = []
            for views in val_views:
                probs.append(predict(model, views).mean(axis=0))
            oof[va] = np.stack(probs)
            print(f"fold {fold} image macro-AUC {macro_auc(y[va], oof[va]):.4f}")
        print(f"image oof macro-AUC {macro_auc(y, oof):.4f}")

        all_x, all_y = [], []
        for index, study_id in enumerate(groups):
            for view in train_frames[study_id]:
                all_x.append(view)
                all_y.append(y[index])
        image_model = build_model()
        train_one(image_model, DataLoader(StudyDataset(all_x, np.stack(all_y)), batch_size=8, shuffle=True))
        torch.save(image_model.state_dict(), ROOT / "resnet18_knee.pt")
    else:
        print("too few local DICOMs for the image model; submission uses label prevalence")

    test = pd.read_csv(DATA / "test.csv")
    test_series = pd.read_csv(DATA / "test_series.csv")
    test_frames = study_frames(test_series, DATA / "test_series")
    rows = []
    for study_id in test["StudyInstanceUID"]:
        views = test_frames.get(str(study_id), [])
        if image_model is not None and views:
            prob = predict(image_model, views).mean(axis=0)
        else:
            prob = prevalence
        rows.append({"StudyInstanceUID": study_id, **{name: float(prob[i]) for i, name in enumerate(LABELS)}})
    out = pd.DataFrame(rows)
    path = ROOT / "submission.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path} rows={len(out)}")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
