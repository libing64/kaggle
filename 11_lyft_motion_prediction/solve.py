#!/usr/bin/env python3
"""Lyft Motion Prediction — ResNet18 raster baseline (l5kit).

Predict 50 future (x,y) offsets for each agent from BEV semantic rasters.
Train on scenes/train.zarr, write competition submission.csv for test.zarr.

Requires the `lyft` conda env (Python 3.9 + l5kit). Example:
  conda run -n lyft python -u 11_lyft_motion_prediction/solve.py
"""

from __future__ import annotations

import os

# l5kit's vendored pb2 files need the pure-Python protobuf impl on newer runtimes.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18

try:
    from torchvision.models import ResNet18_Weights
except ImportError:  # torchvisions < 0.13
    ResNet18_Weights = None  # type: ignore
from tqdm import tqdm

from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import AgentDataset
from l5kit.evaluation import write_pred_csv
from l5kit.geometry import transform_points
from l5kit.rasterization import build_rasterizer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CFG_PATH = ROOT / "agent_motion_config.yaml"
CKPT = ROOT / "resnet18_motion.pt"
SUBMISSION = ROOT / "submission.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cfg() -> Dict:
    with CFG_PATH.open() as handle:
        return yaml.safe_load(handle)


def build_model(cfg: Dict) -> nn.Module:
    if ResNet18_Weights is not None:
        try:
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        except TypeError:
            model = resnet18(pretrained=True)
    else:
        model = resnet18(pretrained=True)
    history = cfg["model_params"]["history_num_frames"]
    num_in = 3 + (history + 1) * 2
    model.conv1 = nn.Conv2d(
        num_in,
        model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False,
    )
    num_targets = 2 * cfg["model_params"]["future_num_frames"]
    model.fc = nn.Linear(model.fc.in_features, num_targets)
    return model.to(DEVICE)


def forward(batch, model, criterion):
    inputs = batch["image"].to(DEVICE)
    avail = batch["target_availabilities"].unsqueeze(-1).to(DEVICE)
    targets = batch["target_positions"].to(DEVICE)
    outputs = model(inputs).reshape(targets.shape)
    loss = criterion(outputs, targets)
    loss = (loss * avail).mean()
    return loss, outputs


def make_loader(cfg: Dict, dm: LocalDataManager, key: str, *, train: bool, mask_path: Path | None = None) -> DataLoader:
    section = "train_data_loader" if train else ("test_data_loader" if "test" in key else "val_data_loader")
    loader_cfg = cfg[section]
    rasterizer = build_rasterizer(cfg, dm)
    zarr = ChunkedDataset(dm.require(key)).open()
    agents_mask = None
    if mask_path is not None and mask_path.exists():
        agents_mask = np.load(mask_path)["arr_0"]
        print(f"loaded agents mask {mask_path} sum={int(agents_mask.sum())}")
    dataset = AgentDataset(cfg, zarr, rasterizer, agents_mask=agents_mask)
    return DataLoader(
        dataset,
        shuffle=loader_cfg.get("shuffle", train),
        batch_size=loader_cfg["batch_size"],
        num_workers=loader_cfg["num_workers"],
        pin_memory=DEVICE.type == "cuda",
    )


def train(cfg: Dict, dm: LocalDataManager) -> nn.Module:
    train_key = cfg["train_data_loader"]["key"]
    # Fall back to sample.zarr when full train is unavailable.
    if not (DATA / train_key).exists() and (DATA / "scenes" / "sample.zarr").exists():
        print("train.zarr missing — using scenes/sample.zarr")
        cfg["train_data_loader"]["key"] = "scenes/sample.zarr"
        train_key = cfg["train_data_loader"]["key"]

    loader = make_loader(cfg, dm, train_key, train=True)
    model = build_model(cfg)
    opt = optim.Adam(model.parameters(), lr=float(cfg["train_params"].get("lr", 1e-4)))
    criterion = nn.MSELoss(reduction="none")
    max_steps = int(cfg["train_params"]["max_num_steps"])
    print(f"device={DEVICE}  steps={max_steps}  train={train_key}  len={len(loader.dataset)}")

    it = iter(loader)
    losses = []
    t0 = time.time()
    model.train()
    pbar = tqdm(range(1, max_steps + 1), desc="train")
    for step in pbar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        loss, _ = forward(batch, model, criterion)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if step % 50 == 0:
            pbar.set_postfix(loss=np.mean(losses[-50:]))
        if step % int(cfg["train_params"]["checkpoint_every_n_steps"]) == 0 or step == max_steps:
            torch.save({"model": model.state_dict(), "cfg": cfg, "step": step}, CKPT)
            print(f"  saved {CKPT} step={step} avg_loss={np.mean(losses[-100:]):.4f}", flush=True)

    print(f"train done seconds={time.time() - t0:.1f}")
    return model


@torch.no_grad()
def predict_test(cfg: Dict, dm: LocalDataManager, model: nn.Module) -> None:
    test_key = cfg["test_data_loader"]["key"]
    if not (DATA / test_key).exists():
        # Local sanity: predict on sample / validate if test absent.
        for fallback in ("scenes/sample.zarr", "scenes/validate.zarr"):
            if (DATA / fallback).exists():
                print(f"test.zarr missing — predicting on {fallback}")
                test_key = fallback
                break
        else:
            raise SystemExit("no zarr dataset found for inference")

    mask_path = None
    if "test.zarr" in test_key:
        # Competition ships mask.npz for the chopped private/public test agents.
        for candidate in [
            DATA / "scenes" / "mask.npz",
            DATA / "mask.npz",
            DATA / "scenes" / "test" / "mask.npz",
        ]:
            if candidate.exists():
                mask_path = candidate
                break

    loader = make_loader(cfg, dm, test_key, train=False, mask_path=mask_path)
    model.eval()
    criterion = nn.MSELoss(reduction="none")

    coords_list, timestamps, track_ids = [], [], []
    for batch in tqdm(loader, desc="predict"):
        _, outputs = forward(batch, model, criterion)
        agents_coords = outputs.cpu().numpy()
        world_from_agents = batch["world_from_agent"].numpy()
        centroids = batch["centroid"].numpy()
        coords_offset = transform_points(agents_coords, world_from_agents) - centroids[:, None, :2]
        coords_list.append(coords_offset)
        timestamps.append(batch["timestamp"].numpy().copy())
        track_ids.append(batch["track_id"].numpy().copy())

    write_pred_csv(
        str(SUBMISSION),
        timestamps=np.concatenate(timestamps),
        track_ids=np.concatenate(track_ids),
        coords=np.concatenate(coords_list),
    )
    print(f"wrote {SUBMISSION}")


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"missing {DATA}; run download.py first")

    os.environ["L5KIT_DATA_FOLDER"] = str(DATA)
    cfg = load_cfg()
    dm = LocalDataManager(None)

    if CKPT.exists():
        print(f"loading {CKPT}")
        payload = torch.load(CKPT, map_location="cpu")
        model = build_model(cfg)
        model.load_state_dict(payload["model"])
    else:
        model = train(cfg, dm)

    predict_test(cfg, dm, model)


if __name__ == "__main__":
    main()
