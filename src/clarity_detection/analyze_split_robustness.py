from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .data import GROUP_MODES, LABEL_COLUMN, answer_key, get_splits, qa_key
from .labels import LABELS
from .metrics import classification_metrics, write_json
from .train_baseline import make_pipeline


def markdown_table(frame: pd.DataFrame) -> str:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare internal split modes for leakage and baseline robustness.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--run-baseline", action="store_true")
    return parser.parse_args()


def leakage_count(left: pd.Series, right: pd.Series) -> tuple[int, int]:
    shared = set(left) & set(right)
    return len(shared), int(right.isin(shared).sum())


def label_counts(frame: pd.DataFrame, prefix: str) -> dict[str, int]:
    counts = frame[LABEL_COLUMN].value_counts().reindex(LABELS, fill_value=0)
    return {f"{prefix}_{label}": int(counts[label]) for label in LABELS}


def evaluate_baseline(splits, output_dir: Path, mode: str) -> dict:
    model = make_pipeline()
    y_train = splits.train["label_id"].to_numpy()
    model.fit(splits.train["text"], y_train)
    dev_pred = model.predict(splits.dev["text"])
    final_pred = model.predict(splits.final["text"])
    dev_metrics = classification_metrics(splits.dev["label_id"].to_numpy(), dev_pred, LABELS)
    final_metrics = classification_metrics(splits.final["label_id"].to_numpy(), final_pred, LABELS)
    mode_dir = output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, mode_dir / "tfidf_baseline.joblib")
    write_json(dev_metrics, mode_dir / "dev_metrics.json")
    write_json(final_metrics, mode_dir / "final_metrics.json")
    return {
        "dev_macro_f1": dev_metrics["macro_f1"],
        "dev_accuracy": dev_metrics["accuracy"],
        "final_macro_f1": final_metrics["macro_f1"],
        "final_accuracy": final_metrics["accuracy"],
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for mode in GROUP_MODES:
        splits = get_splits(seed=args.seed, group_mode=mode)
        qa_shared_groups, qa_dev_rows = leakage_count(qa_key(splits.train), qa_key(splits.dev))
        answer_shared_groups, answer_dev_rows = leakage_count(answer_key(splits.train), answer_key(splits.dev))
        row = {
            "group_mode": mode,
            "train_rows": len(splits.train),
            "dev_rows": len(splits.dev),
            "qa_shared_groups_train_dev": qa_shared_groups,
            "qa_shared_dev_rows": qa_dev_rows,
            "answer_shared_groups_train_dev": answer_shared_groups,
            "answer_shared_dev_rows": answer_dev_rows,
            **label_counts(splits.train, "train"),
            **label_counts(splits.dev, "dev"),
        }
        if args.run_baseline:
            row.update(evaluate_baseline(splits, args.output_dir, mode))
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "split_robustness_summary.csv", index=False)

    lines = [
        "# Split Robustness Analysis",
        "",
        "Lower shared-row counts indicate a stricter internal validation split.",
        "",
        markdown_table(summary),
        "",
        "## Interpretation",
        "",
        "- `url_question_answer` is the original project split mode.",
        "- `qa_text` prevents exact question-answer duplicates across train/dev.",
        "- `answer_text` is the strictest useful mode for this dataset because answer-only duplicates are common and often label-conflicting.",
        "- A large drop from original-split baseline to answer-text baseline means internal-dev scores were partly inflated by repeated answer patterns.",
    ]
    (args.output_dir / "split_robustness_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
