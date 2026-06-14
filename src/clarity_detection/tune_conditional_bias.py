from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .feature_utils import numeric_text_features
from .labels import LABELS
from .metrics import classification_metrics, write_json, write_report
from .plots import plot_confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune simple feature-conditional logit biases on prediction CSVs.")
    parser.add_argument("--dev-predictions", type=Path, required=True)
    parser.add_argument("--final-predictions", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bias-values", nargs="+", type=float, default=[-0.6, -0.3, -0.15, 0.0, 0.15, 0.3, 0.6])
    parser.add_argument("--passes", type=int, default=3)
    return parser.parse_args()


def labels_to_ids(frame: pd.DataFrame) -> np.ndarray:
    label2id = {label: idx for idx, label in enumerate(LABELS)}
    return frame["clarity_label"].map(label2id).to_numpy()


def logits_from_probs(frame: pd.DataFrame) -> np.ndarray:
    probs = frame[[f"prob_{label}" for label in LABELS]].astype(float).to_numpy()
    return np.log(np.clip(probs, 1e-8, 1.0))


def bucket_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    feats = numeric_text_features(frame)
    buckets = {
        "all": np.ones(len(frame), dtype=bool),
        "short_answer": feats["answer_words"].to_numpy() <= 80,
        "very_short_answer": feats["answer_words"].to_numpy() <= 25,
        "long_answer": feats["answer_words"].to_numpy() >= 300,
        "low_overlap": feats["question_answer_overlap"].to_numpy() <= 0.25,
        "high_overlap": feats["question_answer_overlap"].to_numpy() >= 0.55,
        "starts_evasive": feats["starts_with_evasion_marker"].to_numpy() > 0,
        "has_question_back": feats["has_question_back"].to_numpy() > 0,
        "short_low_overlap": (feats["answer_words"].to_numpy() <= 80)
        & (feats["question_answer_overlap"].to_numpy() <= 0.25),
        "long_high_overlap": (feats["answer_words"].to_numpy() >= 300)
        & (feats["question_answer_overlap"].to_numpy() >= 0.55),
    }
    names = list(buckets)
    return np.vstack([buckets[name] for name in names]).T.astype(float), names


def apply_bias(logits: np.ndarray, buckets: np.ndarray, bias_tensor: np.ndarray) -> np.ndarray:
    return logits + buckets @ bias_tensor


def macro_f1(logits: np.ndarray, y_true: np.ndarray, buckets: np.ndarray, bias_tensor: np.ndarray) -> float:
    preds = apply_bias(logits, buckets, bias_tensor).argmax(axis=1)
    return classification_metrics(y_true, preds, LABELS)["macro_f1"]


def tune(logits: np.ndarray, y_true: np.ndarray, buckets: np.ndarray, values: list[float], passes: int) -> tuple[np.ndarray, list[dict]]:
    bias_tensor = np.zeros((buckets.shape[1], len(LABELS)), dtype=np.float32)
    history = []
    best = macro_f1(logits, y_true, buckets, bias_tensor)
    for pass_idx in range(1, passes + 1):
        improved = False
        for bucket_idx in range(buckets.shape[1]):
            if buckets[:, bucket_idx].sum() == 0:
                continue
            for class_idx in range(len(LABELS)):
                current = bias_tensor[bucket_idx, class_idx]
                local_best = best
                local_value = current
                for value in values:
                    bias_tensor[bucket_idx, class_idx] = value
                    score = macro_f1(logits, y_true, buckets, bias_tensor)
                    if score > local_best:
                        local_best = score
                        local_value = value
                bias_tensor[bucket_idx, class_idx] = local_value
                if local_best > best:
                    improved = True
                    best = local_best
                    history.append(
                        {
                            "pass": pass_idx,
                            "bucket_index": bucket_idx,
                            "class_index": class_idx,
                            "bias": float(local_value),
                            "macro_f1": float(best),
                        }
                    )
        if not improved:
            break
    return bias_tensor, history


def evaluate(frame: pd.DataFrame, bias_tensor: np.ndarray, output_dir: Path, split: str) -> dict:
    logits = logits_from_probs(frame)
    y_true = labels_to_ids(frame)
    buckets, _ = bucket_matrix(frame)
    adjusted = apply_bias(logits, buckets, bias_tensor)
    preds = adjusted.argmax(axis=1)
    exp = np.exp(adjusted - adjusted.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    metrics = classification_metrics(y_true, preds, LABELS)
    write_json(metrics, output_dir / f"{split}_metrics.json")
    write_report(metrics, LABELS, output_dir / f"{split}_classification_report.csv")
    pred_frame = frame[["question", "interview_answer", "clarity_label"]].copy()
    pred_frame["prediction"] = [LABELS[idx] for idx in preds]
    for idx, label in enumerate(LABELS):
        pred_frame[f"prob_{label}"] = probs[:, idx]
    pred_frame.to_csv(output_dir / f"{split}_predictions.csv", index=False)
    plot_confusion_matrix(metrics["confusion_matrix"], LABELS, output_dir / f"{split}_confusion.png")
    return metrics


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dev = pd.read_csv(args.dev_predictions)
    logits = logits_from_probs(dev)
    y_true = labels_to_ids(dev)
    buckets, bucket_names = bucket_matrix(dev)
    bias_tensor, history = tune(logits, y_true, buckets, args.bias_values, args.passes)

    bias_rows = []
    for bucket_idx, bucket_name in enumerate(bucket_names):
        for class_idx, label in enumerate(LABELS):
            value = float(bias_tensor[bucket_idx, class_idx])
            if value:
                bias_rows.append({"bucket": bucket_name, "label": label, "bias": value})
    pd.DataFrame(bias_rows).to_csv(args.output_dir / "selected_conditional_biases.csv", index=False)
    pd.DataFrame(history).to_csv(args.output_dir / "tuning_history.csv", index=False)

    summary = {
        "dev_predictions": str(args.dev_predictions),
        "bias_values": args.bias_values,
        "passes": args.passes,
        "selected_biases": bias_rows,
        "dev_fit": evaluate(dev, bias_tensor, args.output_dir, "dev_fit"),
    }
    if args.final_predictions:
        final = pd.read_csv(args.final_predictions)
        summary["final"] = evaluate(final, bias_tensor, args.output_dir, "final")
    write_json(summary, args.output_dir / "conditional_bias_summary.json")


if __name__ == "__main__":
    main()
