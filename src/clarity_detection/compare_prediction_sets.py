from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .labels import LABELS
from .metrics import classification_metrics, write_json


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=PATH")
    name, path = value.split("=", 1)
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple prediction CSVs on the same split.")
    parser.add_argument("--prediction", type=parse_named_path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def label_ids(labels: pd.Series) -> np.ndarray:
    label2id = {label: idx for idx, label in enumerate(LABELS)}
    return labels.map(label2id).to_numpy()


def load_predictions(named_paths: list[tuple[str, Path]]) -> dict[str, pd.DataFrame]:
    frames = {}
    base_keys = None
    for name, path in named_paths:
        frame = pd.read_csv(path)
        keys = frame[["question", "interview_answer", "clarity_label"]].astype(str)
        if base_keys is None:
            base_keys = keys
        elif not keys.equals(base_keys):
            raise ValueError(f"{name} does not align row-for-row with the first prediction file.")
        frames[name] = frame
    return frames


def majority_vote(pred_matrix: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    out = []
    for row, truth in zip(pred_matrix, y_true):
        counts = np.bincount(row, minlength=len(LABELS))
        winners = np.flatnonzero(counts == counts.max())
        if len(winners) == 1:
            out.append(int(winners[0]))
        else:
            # Deterministic tie-breaker for evaluation; not an oracle.
            out.append(int(row[0]))
    return np.asarray(out)


def oracle_predictions(pred_matrix: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    out = []
    for row, truth in zip(pred_matrix, y_true):
        out.append(int(truth) if truth in row else int(row[0]))
    return np.asarray(out)


def write_markdown(path: Path, summary: pd.DataFrame, pairwise: pd.DataFrame, unique: pd.DataFrame) -> None:
    def table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "_No rows._"
        text = frame.copy()
        for col in text.columns:
            if pd.api.types.is_float_dtype(text[col]):
                text[col] = text[col].map(lambda value: f"{value:.4f}")
            else:
                text[col] = text[col].astype(str)
        header = "| " + " | ".join(text.columns) + " |"
        divider = "| " + " | ".join(["---"] * len(text.columns)) + " |"
        rows = ["| " + " | ".join(row) + " |" for row in text.to_numpy(dtype=str)]
        return "\n".join([header, divider, *rows])

    lines = [
        "# Prediction Set Comparison",
        "",
        "## Summary",
        "",
        table(summary),
        "",
        "## Pairwise Complementarity",
        "",
        table(pairwise),
        "",
        "## Unique Correct Counts",
        "",
        table(unique),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = load_predictions(args.prediction)
    names = list(frames)
    y_true = label_ids(next(iter(frames.values()))["clarity_label"])
    pred_ids = {name: label_ids(frame["prediction"]) for name, frame in frames.items()}
    pred_matrix = np.vstack([pred_ids[name] for name in names]).T

    summary_rows = []
    for name in names:
        metrics = classification_metrics(y_true, pred_ids[name], LABELS)
        summary_rows.append(
            {
                "name": name,
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
            }
        )
        write_json(metrics, args.output_dir / f"{name}_metrics.json")

    majority = majority_vote(pred_matrix, y_true)
    oracle = oracle_predictions(pred_matrix, y_true)
    summary_rows.append(
        {
            "name": "majority_vote",
            "macro_f1": f1_score(y_true, majority, average="macro"),
            "accuracy": float(np.mean(majority == y_true)),
        }
    )
    summary_rows.append(
        {
            "name": "oracle_if_any_model_correct",
            "macro_f1": f1_score(y_true, oracle, average="macro"),
            "accuracy": float(np.mean(oracle == y_true)),
        }
    )
    summary = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False)

    pair_rows = []
    for idx, left in enumerate(names):
        for right in names[idx + 1 :]:
            left_correct = pred_ids[left] == y_true
            right_correct = pred_ids[right] == y_true
            pair_rows.append(
                {
                    "left": left,
                    "right": right,
                    "disagreement_rate": float(np.mean(pred_ids[left] != pred_ids[right])),
                    "left_only_correct": int(np.sum(left_correct & ~right_correct)),
                    "right_only_correct": int(np.sum(right_correct & ~left_correct)),
                    "both_wrong": int(np.sum(~left_correct & ~right_correct)),
                }
            )
    pairwise = pd.DataFrame(pair_rows).sort_values("disagreement_rate", ascending=False)

    unique_rows = []
    correctness = np.vstack([pred_ids[name] == y_true for name in names]).T
    for col, name in enumerate(names):
        unique_rows.append(
            {
                "name": name,
                "unique_correct": int(np.sum(correctness[:, col] & (correctness.sum(axis=1) == 1))),
                "correct": int(np.sum(correctness[:, col])),
            }
        )
    unique = pd.DataFrame(unique_rows).sort_values("unique_correct", ascending=False)

    summary.to_csv(args.output_dir / "summary.csv", index=False)
    pairwise.to_csv(args.output_dir / "pairwise.csv", index=False)
    unique.to_csv(args.output_dir / "unique_correct.csv", index=False)
    write_markdown(args.output_dir / "comparison.md", summary, pairwise, unique)


if __name__ == "__main__":
    main()
