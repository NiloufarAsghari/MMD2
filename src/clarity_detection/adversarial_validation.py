from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline

from .data import LABEL_COLUMN, load_qevasion


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
    parser = argparse.ArgumentParser(description="Adversarial validation between official train and final text.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def combined_text(frame: pd.DataFrame) -> pd.Series:
    return (
        "Question: "
        + frame["question"].fillna("").astype(str)
        + "\nAnswer: "
        + frame["interview_answer"].fillna("").astype(str)
    )


def make_model(seed: int) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                    max_features=60_000,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="char_wb",
                    ngram_range=(4, 6),
                    min_df=2,
                    sublinear_tf=True,
                    max_features=60_000,
                ),
            ),
        ]
    )
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=seed)
    return Pipeline([("features", features), ("clf", clf)])


def feature_names(model: Pipeline) -> np.ndarray:
    union = model.named_steps["features"]
    names = []
    for name, transformer in union.transformer_list:
        names.extend(f"{name}:{feature}" for feature in transformer.get_feature_names_out())
    return np.asarray(names)


def top_coefficients(model: Pipeline, n: int = 40) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = feature_names(model)
    coef = model.named_steps["clf"].coef_[0]
    order = np.argsort(coef)
    train_like = pd.DataFrame({"feature": names[order[:n]], "coefficient": coef[order[:n]]})
    final_like = pd.DataFrame({"feature": names[order[-n:]][::-1], "coefficient": coef[order[-n:]][::-1]})
    return train_like, final_like


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train, final = load_qevasion()
    train = train.copy()
    final = final.copy()
    train["source_split"] = "official_train"
    final["source_split"] = "official_final"
    frame = pd.concat([train, final], ignore_index=True)
    x = combined_text(frame)
    y = (frame["source_split"] == "official_final").astype(int).to_numpy()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    oof_scores = cross_val_predict(make_model(args.seed), x, y, cv=cv, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, oof_scores))
    frame["final_like_score_oof"] = oof_scores
    frame[["source_split", LABEL_COLUMN, "question", "interview_answer", "final_like_score_oof"]].to_csv(
        args.output_dir / "adversarial_oof_scores.csv",
        index=False,
    )

    model = make_model(args.seed)
    model.fit(x, y)
    train_like, final_like = top_coefficients(model)
    train_like.to_csv(args.output_dir / "top_train_like_features.csv", index=False)
    final_like.to_csv(args.output_dir / "top_final_like_features.csv", index=False)

    source_summary = frame.groupby("source_split")["final_like_score_oof"].agg(["count", "mean", "median", "std"])
    label_summary = (
        frame.groupby(["source_split", LABEL_COLUMN])["final_like_score_oof"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    source_summary.to_csv(args.output_dir / "source_score_summary.csv")
    label_summary.to_csv(args.output_dir / "label_score_summary.csv", index=False)

    lines = [
        "# Adversarial Validation",
        "",
        f"Train-vs-final text AUC: `{auc:.6f}`",
        "",
        "AUC near 0.5 means the final split looks like train text. Higher AUC means distribution shift.",
        "",
        "## Source Score Summary",
        "",
        markdown_table(source_summary.reset_index()),
        "",
        "## Label Score Summary",
        "",
        markdown_table(label_summary),
        "",
        "## Top Final-Like Features",
        "",
        markdown_table(final_like.head(20)),
        "",
        "## Top Train-Like Features",
        "",
        markdown_table(train_like.head(20)),
    ]
    (args.output_dir / "adversarial_validation_report.md").write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame([{"auc": auc, "rows": len(frame), "train_rows": len(train), "final_rows": len(final)}]).to_csv(
        args.output_dir / "adversarial_validation_summary.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
