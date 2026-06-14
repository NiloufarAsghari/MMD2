from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import LABEL_COLUMN, get_splits
from .feature_utils import numeric_text_features
from .labels import LABELS
from .metrics import classification_metrics, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze QEvasion data and existing prediction errors.")
    parser.add_argument("--dev-predictions", type=Path, required=True)
    parser.add_argument("--final-predictions", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=13)
    return parser.parse_args()


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = numeric_text_features(frame)
    return pd.concat([frame.reset_index(drop=True), features.reset_index(drop=True)], axis=1)


def label_distribution(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    counts = frame[LABEL_COLUMN].value_counts().reindex(LABELS, fill_value=0)
    return pd.DataFrame(
        {
            "split": split,
            "label": counts.index,
            "count": counts.values,
            "share": counts.values / max(1, len(frame)),
        }
    )


def feature_summary(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    feature_cols = [
        "question_words",
        "answer_words",
        "answer_chars",
        "answer_sentences",
        "question_answer_overlap",
        "answer_question_overlap",
        "answer_to_question_len",
        "direct_marker_count",
        "evasion_marker_count",
        "starts_with_direct_marker",
        "starts_with_evasion_marker",
        "has_question_back",
    ]
    rows = []
    for label, part in frame.groupby(LABEL_COLUMN):
        row = {"split": split, "label": label, "rows": len(part)}
        for col in feature_cols:
            row[f"{col}_mean"] = float(part[col].mean())
            row[f"{col}_median"] = float(part[col].median())
        rows.append(row)
    return pd.DataFrame(rows)


def error_summary(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    if "prediction" not in frame.columns:
        return pd.DataFrame()
    rows = []
    for (truth, pred), part in frame.groupby([LABEL_COLUMN, "prediction"]):
        rows.append(
            {
                "split": split,
                "truth": truth,
                "prediction": pred,
                "count": len(part),
                "mean_answer_words": float(part["answer_words"].mean()),
                "mean_prob_max": float(part[[col for col in part.columns if col.startswith("prob_")]].astype(float).max(axis=1).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "truth", "count"], ascending=[True, True, False])


def high_confidence_errors(frame: pd.DataFrame, split: str, n: int = 30) -> pd.DataFrame:
    if "prediction" not in frame.columns:
        return pd.DataFrame()
    prob_cols = [col for col in frame.columns if col.startswith("prob_")]
    errors = frame[frame[LABEL_COLUMN] != frame["prediction"]].copy()
    if errors.empty:
        return errors
    errors["split"] = split
    errors["prob_max"] = errors[prob_cols].astype(float).max(axis=1)
    cols = [
        "split",
        "question",
        "interview_answer",
        LABEL_COLUMN,
        "prediction",
        "prob_max",
        "answer_words",
        "question_answer_overlap",
        "direct_marker_count",
        "evasion_marker_count",
    ]
    return errors.sort_values("prob_max", ascending=False)[cols].head(n)


def write_markdown(
    path: Path,
    *,
    split_rows: pd.DataFrame,
    metrics: dict[str, dict],
    feature_rows: pd.DataFrame,
    error_rows: pd.DataFrame,
    notes: list[str],
) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "_No rows._"
        text_frame = frame.copy()
        for col in text_frame.columns:
            if pd.api.types.is_float_dtype(text_frame[col]):
                text_frame[col] = text_frame[col].map(lambda value: f"{value:.3f}")
            else:
                text_frame[col] = text_frame[col].astype(str)
        header = "| " + " | ".join(text_frame.columns) + " |"
        divider = "| " + " | ".join(["---"] * len(text_frame.columns)) + " |"
        rows = ["| " + " | ".join(row) + " |" for row in text_frame.to_numpy(dtype=str)]
        return "\n".join([header, divider, *rows])

    lines = ["# QEvasion Dataset and Error Analysis", ""]
    lines.append("## Label Distribution")
    lines.append("")
    lines.append(markdown_table(split_rows))
    lines.append("")
    lines.append("## Prediction Metrics")
    lines.append("")
    for split, split_metrics in metrics.items():
        lines.append(f"- `{split}` Macro-F1: `{split_metrics['macro_f1']:.6f}`, accuracy: `{split_metrics['accuracy']:.6f}`")
    lines.append("")
    lines.append("## Feature Summary by Label")
    lines.append("")
    keep_cols = [
        "split",
        "label",
        "rows",
        "answer_words_mean",
        "answer_words_median",
        "question_answer_overlap_mean",
        "direct_marker_count_mean",
        "evasion_marker_count_mean",
        "starts_with_evasion_marker_mean",
    ]
    lines.append(markdown_table(feature_rows[keep_cols]))
    lines.append("")
    lines.append("## Error Counts")
    lines.append("")
    if error_rows.empty:
        lines.append("No prediction errors available.")
    else:
        lines.append(markdown_table(error_rows))
    lines.append("")
    lines.append("## Modeling Notes")
    lines.append("")
    lines.extend(f"- {note}" for note in notes)
    path.write_text("\n".join(lines), encoding="utf-8")


def metrics_for(frame: pd.DataFrame) -> dict:
    y_true = frame[LABEL_COLUMN].map({label: idx for idx, label in enumerate(LABELS)}).to_numpy()
    y_pred = frame["prediction"].map({label: idx for idx, label in enumerate(LABELS)}).to_numpy()
    return classification_metrics(y_true, y_pred, LABELS)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits = get_splits(seed=args.split_seed)
    train = add_features(splits.train)
    dev_pred = add_features(pd.read_csv(args.dev_predictions))
    frames = {"train": train, "dev_pred": dev_pred}
    if args.final_predictions:
        frames["final_pred"] = add_features(pd.read_csv(args.final_predictions))
    else:
        frames["final"] = add_features(splits.final)

    split_rows = pd.concat(
        [label_distribution(frame, split) for split, frame in frames.items()],
        ignore_index=True,
    )
    feature_rows = pd.concat(
        [feature_summary(frame, split) for split, frame in frames.items()],
        ignore_index=True,
    )
    error_rows = pd.concat(
        [error_summary(frame, split) for split, frame in frames.items()],
        ignore_index=True,
    )
    high_conf_rows = pd.concat(
        [high_confidence_errors(frame, split) for split, frame in frames.items()],
        ignore_index=True,
    )

    metrics = {
        split: metrics_for(frame)
        for split, frame in frames.items()
        if "prediction" in frame.columns
    }
    for split, split_metrics in metrics.items():
        write_json(split_metrics, args.output_dir / f"{split}_metrics.json")

    split_rows.to_csv(args.output_dir / "label_distribution.csv", index=False)
    feature_rows.to_csv(args.output_dir / "feature_summary_by_label.csv", index=False)
    error_rows.to_csv(args.output_dir / "error_summary.csv", index=False)
    high_conf_rows.to_csv(args.output_dir / "high_confidence_errors.csv", index=False)

    notes = [
        "This report uses only question/answer text plus existing model predictions.",
        "Final prediction analysis is diagnostic if final labels have already been inspected.",
        "Large answer length and weak lexical overlap are the first checks for long-context or hierarchical models.",
        "High-confidence Clear Reply/Ambivalent flips indicate the main model often mistakes short direct-sounding answers for actually responsive answers.",
    ]
    write_markdown(
        args.output_dir / "analysis.md",
        split_rows=split_rows,
        metrics=metrics,
        feature_rows=feature_rows,
        error_rows=error_rows,
        notes=notes,
    )


if __name__ == "__main__":
    main()
