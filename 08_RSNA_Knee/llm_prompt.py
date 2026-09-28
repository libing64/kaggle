#!/usr/bin/env python3
"""Zero-shot knee-report labeling with a local Qwen prompt.

The hidden test set has no radiology report. This scores the 58 labeled
training reports and writes one JSON probability object per study.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "rsna-knee"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
CACHE = ROOT / "llm_cache.jsonl"
OUT = ROOT / "llm_labeled_predictions.csv"
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
GUIDE = {
    "ACL": "anterior cruciate ligament tear or sprain",
    "MCL": "medial collateral ligament tear or sprain",
    "Medial Meniscus": "medial meniscus tear",
    "Lateral Meniscus": "lateral meniscus tear",
    "Medial OA": "medial tibiofemoral osteoarthritis or cartilage loss",
    "Lateral OA": "lateral tibiofemoral osteoarthritis or cartilage loss",
    "PF OA": "patellofemoral osteoarthritis or cartilage loss",
    "Effusion": "joint effusion or excess fluid",
    "Synovitis": "synovitis",
    "Baker's": "Baker's cyst or popliteal cyst",
    "Contusion": "bone contusion or bone bruise",
    "Fracture": "fracture",
}


def build_prompt(report: str) -> str:
    lines = "\n".join(f"- {name}: {GUIDE[name]}" for name in LABELS)
    keys = ", ".join(f'"{name}": 0.0' for name in LABELS)
    return (
        "You extract findings from a knee MRI radiology report. "
        "The report may be in any language. "
        "For each finding, give the probability it is present, from 0 to 1. "
        "Use a value near 1 only when the report affirms the finding, "
        "and near 0 when it is denied or not mentioned.\n"
        f"Findings:\n{lines}\n"
        "Reply with one JSON object and nothing else. Keys must match exactly:\n"
        f"{{{keys}}}\n\n"
        f"Report:\n{report.strip()}"
    )


def parse_scores(text: str) -> dict[str, float]:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("no json")
    raw = json.loads(match.group(0))
    scores = {}
    for name in LABELS:
        value = raw.get(name, raw.get(name.lower(), 0.5))
        scores[name] = float(np.clip(float(value), 0.0, 1.0))
    return scores


def load_cache() -> dict[str, dict[str, float]]:
    cache = {}
    if not CACHE.exists():
        return cache
    for line in CACHE.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cache[row["StudyInstanceUID"]] = row["scores"]
    return cache


def macro_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    scores = []
    for col in range(y_true.shape[1]):
        if len(np.unique(y_true[:, col])) < 2:
            continue
        scores.append(roc_auc_score(y_true[:, col], y_prob[:, col]))
    return float(np.mean(scores)) if scores else float("nan")


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    labeled = train.dropna(subset=LABELS).reset_index(drop=True)
    cache = load_cache()
    pending = labeled[~labeled["StudyInstanceUID"].isin(cache)]
    print(f"labeled={len(labeled)} cached={len(labeled) - len(pending)} pending={len(pending)}")

    if len(pending):
        processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).to("cuda")
        model.eval()
        with CACHE.open("a") as handle:
            for index, row in enumerate(pending.itertuples(index=False), start=1):
                messages = [{"role": "user", "content": [{"type": "text", "text": build_prompt(row.Report)}]}]
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[prompt], return_tensors="pt").to(model.device)
                with torch.no_grad():
                    output = model.generate(**inputs, max_new_tokens=220, do_sample=False)
                generated = output[0, inputs["input_ids"].shape[1] :]
                text = processor.decode(generated, skip_special_tokens=True)
                try:
                    scores = parse_scores(text)
                except Exception:
                    scores = {name: 0.5 for name in LABELS}
                    text = "PARSE_FAIL " + text[:200]
                record = {"StudyInstanceUID": row.StudyInstanceUID, "scores": scores, "raw": text}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                cache[row.StudyInstanceUID] = scores
                if index == 1 or index % 10 == 0 or index == len(pending):
                    print(f"{index}/{len(pending)} {row.StudyInstanceUID[-8:]} {text[:120].replace(chr(10), ' ')}")

    probs = np.array([[cache[uid][name] for name in LABELS] for uid in labeled["StudyInstanceUID"]], dtype=np.float32)
    y = labeled[LABELS].to_numpy(dtype=np.float32)
    print(f"prompt macro-AUC {macro_auc(y, probs):.4f}")
    per_label = []
    for col, name in enumerate(LABELS):
        if len(np.unique(y[:, col])) < 2:
            continue
        per_label.append((name, roc_auc_score(y[:, col], probs[:, col])))
    for name, score in sorted(per_label, key=lambda item: item[1], reverse=True):
        print(f"  {name:20s} {score:.3f}")

    out = labeled[["StudyInstanceUID"]].copy()
    for col, name in enumerate(LABELS):
        out[name] = probs[:, col]
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
