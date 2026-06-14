from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import LABEL_COLUMN, stable_text_key


KEY_COLUMNS = ["question", "interview_answer", LABEL_COLUMN]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build external train sample weights for transformer runs.")
    parser.add_argument("--score-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--score-column", default="final_like_score_oof")
    parser.add_argument("--source-split", default="official_train")
    parser.add_argument("--floor", type=float, default=0.5)
    parser.add_argument("--power", type=float, default=1.0)
    parser.add_argument("--max-weight", type=float, default=3.0)
    parser.add_argument("--conflict-key", choices=["none", "qa", "answer"], default="qa")
    parser.add_argument("--conflict-downweight", type=float, default=0.75)
    return parser.parse_args()


def conflict_keys(frame: pd.DataFrame, mode: str) -> set[str]:
    if mode == "none":
        return set()
    if mode == "qa":
        keys = [stable_text_key(q, a) for q, a in zip(frame["question"], frame["interview_answer"])]
    elif mode == "answer":
        keys = [stable_text_key(a) for a in frame["interview_answer"]]
    else:
        raise ValueError(f"Unknown conflict mode: {mode}")
    tmp = pd.DataFrame({"key": keys, "label": frame[LABEL_COLUMN].to_numpy()})
    return set(tmp.groupby("key")["label"].nunique().loc[lambda values: values > 1].index)


def row_keys(frame: pd.DataFrame, mode: str) -> list[str]:
    if mode == "qa":
        return [stable_text_key(q, a) for q, a in zip(frame["question"], frame["interview_answer"])]
    if mode == "answer":
        return [stable_text_key(a) for a in frame["interview_answer"]]
    return [""] * len(frame)


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.score_csv)
    missing_cols = [col for col in [*KEY_COLUMNS, "source_split", args.score_column] if col not in frame.columns]
    if missing_cols:
        raise ValueError(f"Score CSV is missing columns: {missing_cols}")
    frame = frame[frame["source_split"].eq(args.source_split)].copy()
    if frame.empty:
        raise ValueError(f"No rows found for source_split={args.source_split!r}")

    score = frame[args.score_column].astype(float).clip(lower=0.0).to_numpy()
    weights = args.floor + np.power(score, args.power)
    conflict_set = conflict_keys(frame, args.conflict_key)
    if conflict_set:
        row_key_values = row_keys(frame, args.conflict_key)
        conflict_mask = np.asarray([key in conflict_set for key in row_key_values], dtype=bool)
        weights[conflict_mask] *= args.conflict_downweight
    else:
        conflict_mask = np.zeros(len(frame), dtype=bool)

    weights = weights / max(float(weights.mean()), 1e-12)
    weights = np.clip(weights, 0.0, args.max_weight)
    weights = weights / max(float(weights.mean()), 1e-12)

    out = frame[KEY_COLUMNS + ["source_split", args.score_column]].copy()
    out["sample_weight"] = weights
    out["conflict_downweighted"] = conflict_mask
    out.to_csv(args.output_csv, index=False)

    summary = pd.DataFrame(
        [
            {
                "rows": len(out),
                "score_column": args.score_column,
                "source_split": args.source_split,
                "floor": args.floor,
                "power": args.power,
                "max_weight": args.max_weight,
                "conflict_key": args.conflict_key,
                "conflict_downweight": args.conflict_downweight,
                "conflict_rows": int(conflict_mask.sum()),
                "weight_min": float(out["sample_weight"].min()),
                "weight_mean": float(out["sample_weight"].mean()),
                "weight_max": float(out["sample_weight"].max()),
            }
        ]
    )
    summary.to_csv(args.output_csv.with_suffix(".summary.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
