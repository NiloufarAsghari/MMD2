from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from .data import LABEL_COLUMN, get_splits, load_qevasion
from .labels import LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze duplicate and ambiguous labels in QEvasion.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=13)
    parser.add_argument("--predictions", type=Path)
    return parser.parse_args()


def stable_text_key(*parts: str) -> str:
    raw = "\n".join(str(part or "").lower().strip() for part in parts)
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()


def add_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["qa_key"] = [stable_text_key(q, a) for q, a in zip(out["question"], out["interview_answer"])]
    out["answer_key"] = [stable_text_key(a) for a in out["interview_answer"]]
    out["question_key"] = [stable_text_key(q) for q in out["question"]]
    return out


def ambiguity_by_key(frame: pd.DataFrame, key: str, split: str) -> pd.DataFrame:
    rows = []
    for key_value, group in frame.groupby(key):
        counts = group[LABEL_COLUMN].value_counts().reindex(LABELS, fill_value=0)
        if len(group) <= 1:
            continue
        rows.append(
            {
                "split": split,
                "key_type": key,
                "key": key_value,
                "rows": len(group),
                "unique_labels": int((counts > 0).sum()),
                "majority_label": counts.idxmax(),
                "majority_share": float(counts.max() / len(group)),
                **{f"count_{label}": int(counts[label]) for label in LABELS},
                "example_question": group["question"].iloc[0],
                "example_answer": group["interview_answer"].iloc[0][:500],
            }
        )
    return pd.DataFrame(rows)


def split_summary(frame: pd.DataFrame, split: str) -> dict:
    summary = {"split": split, "rows": len(frame)}
    for key in ["qa_key", "answer_key", "question_key"]:
        grouped = frame.groupby(key)[LABEL_COLUMN].nunique()
        summary[f"{key}_duplicate_groups"] = int((frame.groupby(key).size() > 1).sum())
        summary[f"{key}_conflicting_groups"] = int((grouped > 1).sum())
        summary[f"{key}_conflicting_rows"] = int(frame[frame[key].isin(grouped[grouped > 1].index)].shape[0])
    return summary


def prediction_ambiguity(predictions: pd.DataFrame, ambiguity: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or ambiguity.empty:
        return pd.DataFrame()
    preds = add_keys(predictions)
    rows = []
    for key_type in ["qa_key", "answer_key", "question_key"]:
        if key_type not in preds.columns:
            continue
        ambiguous_keys = set(
            ambiguity.loc[
                (ambiguity["unique_labels"] > 1) & (ambiguity["key_type"] == key_type),
                "key",
            ]
        )
        subset = preds[preds[key_type].isin(ambiguous_keys)]
        if subset.empty:
            continue
        rows.append(
            {
                "key_type": key_type,
                "rows": len(subset),
                "accuracy": float((subset[LABEL_COLUMN] == subset["prediction"]).mean())
                if "prediction" in subset.columns
                else None,
                "error_rows": int((subset[LABEL_COLUMN] != subset["prediction"]).sum())
                if "prediction" in subset.columns
                else None,
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    text = frame.copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            text[col] = text[col].map(lambda value: f"{value:.3f}")
        else:
            text[col] = text[col].astype(str)
    header = "| " + " | ".join(text.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(text.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in text.to_numpy(dtype=str)]
    return "\n".join([header, divider, *rows])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    official_train, official_final = load_qevasion()
    splits = get_splits(seed=args.split_seed)
    frames = {
        "official_train": add_keys(official_train),
        "internal_train": add_keys(splits.train),
        "internal_dev": add_keys(splits.dev),
        "official_final": add_keys(official_final),
    }

    summary = pd.DataFrame([split_summary(frame, split) for split, frame in frames.items()])
    ambiguity_frames = []
    for split, frame in frames.items():
        for key in ["qa_key", "answer_key", "question_key"]:
            ambiguity_frames.append(ambiguity_by_key(frame, key, split))
    ambiguity = pd.concat(ambiguity_frames, ignore_index=True)
    conflicts = ambiguity[ambiguity["unique_labels"] > 1].sort_values(
        ["split", "key_type", "rows"], ascending=[True, True, False]
    )

    summary.to_csv(args.output_dir / "ambiguity_summary.csv", index=False)
    ambiguity.to_csv(args.output_dir / "duplicate_groups.csv", index=False)
    conflicts.to_csv(args.output_dir / "conflicting_duplicate_groups.csv", index=False)

    pred_summary = pd.DataFrame()
    if args.predictions:
        pred_summary = prediction_ambiguity(pd.read_csv(args.predictions), conflicts)
        pred_summary.to_csv(args.output_dir / "prediction_ambiguity_summary.csv", index=False)

    top_cols = [
        "split",
        "key_type",
        "rows",
        "unique_labels",
        "majority_label",
        "majority_share",
        "count_Clear Reply",
        "count_Ambivalent",
        "count_Clear Non-Reply",
        "example_question",
        "example_answer",
    ]
    lines = [
        "# QEvasion Label Ambiguity Analysis",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## Conflicting Duplicate Groups",
        "",
        markdown_table(conflicts[top_cols].head(30) if not conflicts.empty else conflicts),
        "",
    ]
    if not pred_summary.empty:
        lines.extend(["## Prediction Rows in Ambiguous Groups", "", markdown_table(pred_summary), ""])
    lines.extend(
        [
            "## Interpretation",
            "",
            "- Exact question-answer conflicts are direct label-noise candidates.",
            "- Answer-only conflicts indicate that the same political answer can be clear or evasive depending on the question.",
            "- Question-only conflicts are expected and mostly describe topic diversity, not label noise.",
        ]
    )
    (args.output_dir / "ambiguity_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
