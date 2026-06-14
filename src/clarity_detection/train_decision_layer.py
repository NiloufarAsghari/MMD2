from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .feature_utils import build_decision_features, numeric_text_features, probability_features
from .labels import LABELS
from .metrics import classification_metrics, write_json, write_report
from .plots import plot_confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight decision layer on existing predictions.")
    parser.add_argument("--dev-predictions", type=Path, required=True)
    parser.add_argument("--final-predictions", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--c-grid", nargs="+", type=float, default=[0.03, 0.1, 0.3, 1.0, 3.0])
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--feature-set", choices=["all", "prob", "text"], default="all")
    return parser.parse_args()


def make_features(frame: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    if feature_set == "prob":
        return probability_features(frame)
    if feature_set == "text":
        return numeric_text_features(frame)
    return build_decision_features(frame)


def labels_to_ids(frame: pd.DataFrame) -> np.ndarray:
    label2id = {label: idx for idx, label in enumerate(LABELS)}
    return frame["clarity_label"].map(label2id).to_numpy()


def model_for(c_value: float, class_weight: str) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced" if class_weight == "balanced" else None,
                    max_iter=5000,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def tune_c(x: pd.DataFrame, y: np.ndarray, *, c_grid: list[float], folds: int, class_weight: str) -> pd.DataFrame:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=13)
    rows = []
    for c_value in c_grid:
        fold_scores = []
        for train_idx, valid_idx in splitter.split(x, y):
            model = model_for(c_value, class_weight)
            model.fit(x.iloc[train_idx], y[train_idx])
            preds = model.predict(x.iloc[valid_idx])
            fold_scores.append(f1_score(y[valid_idx], preds, average="macro"))
        rows.append(
            {
                "C": c_value,
                "mean_macro_f1": float(np.mean(fold_scores)),
                "std_macro_f1": float(np.std(fold_scores)),
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_macro_f1", "C"], ascending=[False, True])


def write_predictions(frame: pd.DataFrame, probs: np.ndarray, preds: np.ndarray, path: Path) -> None:
    out = frame[["question", "interview_answer", "clarity_label"]].copy()
    out["prediction"] = [LABELS[idx] for idx in preds]
    for idx, label in enumerate(LABELS):
        out[f"prob_{label}"] = probs[:, idx]
    out.to_csv(path, index=False)


def evaluate_split(model: Pipeline, frame: pd.DataFrame, output_dir: Path, split: str, feature_set: str) -> dict:
    x = make_features(frame, feature_set)
    y = labels_to_ids(frame)
    probs = model.predict_proba(x)
    preds = probs.argmax(axis=1)
    metrics = classification_metrics(y, preds, LABELS)
    write_json(metrics, output_dir / f"{split}_metrics.json")
    write_report(metrics, LABELS, output_dir / f"{split}_classification_report.csv")
    write_predictions(frame, probs, preds, output_dir / f"{split}_predictions.csv")
    plot_confusion_matrix(metrics["confusion_matrix"], LABELS, output_dir / f"{split}_confusion.png")
    return metrics


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dev = pd.read_csv(args.dev_predictions)
    x_dev = make_features(dev, args.feature_set)
    y_dev = labels_to_ids(dev)
    cv_results = tune_c(
        x_dev,
        y_dev,
        c_grid=args.c_grid,
        folds=args.cv_folds,
        class_weight=args.class_weight,
    )
    cv_results.to_csv(args.output_dir / "cv_results.csv", index=False)
    best_c = float(cv_results.iloc[0]["C"])
    model = model_for(best_c, args.class_weight)
    model.fit(x_dev, y_dev)

    metrics = {
        "selected_C": best_c,
        "cv_best_macro_f1": float(cv_results.iloc[0]["mean_macro_f1"]),
        "cv_best_std": float(cv_results.iloc[0]["std_macro_f1"]),
        "class_weight": args.class_weight,
        "feature_set": args.feature_set,
        "dev_predictions": str(args.dev_predictions),
        "features": list(x_dev.columns),
    }
    metrics["dev_fit"] = evaluate_split(model, dev, args.output_dir, "dev_fit", args.feature_set)
    if args.final_predictions:
        final = pd.read_csv(args.final_predictions)
        metrics["final"] = evaluate_split(model, final, args.output_dir, "final", args.feature_set)
    write_json(metrics, args.output_dir / "decision_layer_summary.json")


if __name__ == "__main__":
    main()
