from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion, Pipeline

from .data import GROUP_MODES, LABEL_COLUMN, get_splits, save_split_summary
from .labels import LABEL2ID, LABELS
from .metrics import classification_metrics, write_json, write_report
from .plots import plot_confusion_matrix, plot_label_distribution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TF-IDF sanity baseline.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tfidf_baseline"))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--group-mode", choices=GROUP_MODES, default="url_question_answer")
    parser.add_argument("--eval-final", action="store_true")
    return parser.parse_args()


def make_pipeline() -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    max_features=80_000,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                    max_features=40_000,
                ),
            ),
        ]
    )
    clf = LinearSVC(
        C=0.8,
        class_weight="balanced",
        max_iter=5000,
        random_state=13,
    )
    return Pipeline([("features", features), ("clf", clf)])


def scores_to_prob_like(scores: np.ndarray) -> np.ndarray:
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    scores = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / exp.sum(axis=1, keepdims=True)


def evaluate_split(model: Pipeline, frame: pd.DataFrame, split_name: str, output_dir: Path) -> dict:
    pred = model.predict(frame["text"])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(frame["text"])
    else:
        probs = scores_to_prob_like(model.decision_function(frame["text"]))
    y_true = frame["label_id"].to_numpy()
    metrics = classification_metrics(y_true, pred, LABELS)

    pred_frame = frame[["question", "interview_answer", LABEL_COLUMN]].copy()
    pred_frame["prediction"] = [LABELS[i] for i in pred]
    for idx, label in enumerate(LABELS):
        pred_frame[f"prob_{label}"] = probs[:, idx]
    pred_frame.to_csv(output_dir / f"{split_name}_predictions.csv", index=False)

    write_json(metrics, output_dir / f"{split_name}_metrics.json")
    write_report(metrics, LABELS, output_dir / f"{split_name}_classification_report.csv")
    plot_confusion_matrix(metrics["confusion_matrix"], LABELS, output_dir / f"{split_name}_confusion.png")
    return metrics


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = get_splits(seed=args.seed, group_mode=args.group_mode)
    save_split_summary(splits, args.output_dir)
    plot_label_distribution(splits.train, LABEL_COLUMN, args.output_dir / "train_label_distribution.png")
    plot_label_distribution(splits.dev, LABEL_COLUMN, args.output_dir / "dev_label_distribution.png")

    model = make_pipeline()
    y_train = splits.train[LABEL_COLUMN].map(LABEL2ID).to_numpy()
    model.fit(splits.train["text"], y_train)
    joblib.dump(model, args.output_dir / "tfidf_logreg.joblib")

    dev_metrics = evaluate_split(model, splits.dev, "dev", args.output_dir)
    print(f"dev_macro_f1={dev_metrics['macro_f1']:.6f}")

    if args.eval_final:
        final_metrics = evaluate_split(model, splits.final, "final", args.output_dir)
        print(f"final_macro_f1={final_metrics['macro_f1']:.6f}")


if __name__ == "__main__":
    main()
