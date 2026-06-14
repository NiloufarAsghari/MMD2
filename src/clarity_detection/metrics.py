from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def classification_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    labels: list[str],
) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        labels=list(range(len(labels))),
        output_dict=True,
        zero_division=0,
    )
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float((y_true == y_pred).mean()),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(labels)))
        ).tolist(),
    }


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_report(metrics: dict, labels: list[str], path: Path) -> None:
    report = pd.DataFrame(metrics["classification_report"]).transpose()
    report.to_csv(path, index=True)
    text_path = path.with_suffix(".txt")
    with text_path.open("w", encoding="utf-8") as f:
        f.write(f"Macro-F1: {metrics['macro_f1']:.6f}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.6f}\n\n")
        f.write("Labels:\n")
        for i, label in enumerate(labels):
            f.write(f"{i}: {label}\n")

