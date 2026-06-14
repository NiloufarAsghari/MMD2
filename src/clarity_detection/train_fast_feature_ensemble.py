from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.special import softmax
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from .data import LABEL_COLUMN, load_qevasion
from .labels import LABELS
from .metrics import classification_metrics, write_json, write_report
from .plots import plot_confusion_matrix
from .train_feature_ensemble import HandcraftedFeatures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast cached sparse feature ensemble.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fast_feature_ensemble"))
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-selected", type=int, default=4)
    parser.add_argument("--eval-final", action="store_true")
    return parser.parse_args()


@dataclass(frozen=True)
class Candidate:
    name: str
    profile: str
    estimator: object


@dataclass
class FeaturePack:
    profile: str
    vectorizers: list[tuple[str, str, TfidfVectorizer]]
    hand: HandcraftedFeatures
    scaler: StandardScaler

    def transform(self, frame: pd.DataFrame):
        parts = []
        for _, column, vectorizer in self.vectorizers:
            parts.append(vectorizer.transform(select_text(frame, column)))
        hand = self.hand.transform(frame)
        parts.append(csr_matrix(self.scaler.transform(hand)))
        return hstack(parts, format="csr")


def candidates(seed: int) -> list[Candidate]:
    return [
        Candidate("rich_ridge_a1.0", "rich", RidgeClassifier(alpha=1.0, class_weight="balanced")),
        Candidate("rich_ridge_a3.0", "rich", RidgeClassifier(alpha=3.0, class_weight="balanced")),
        Candidate(
            "rich_sgd_log",
            "rich",
            SGDClassifier(
                loss="log_loss",
                penalty="elasticnet",
                alpha=1e-5,
                l1_ratio=0.15,
                class_weight="balanced",
                max_iter=1200,
                tol=1e-3,
                random_state=seed,
            ),
        ),
        Candidate("answer_ridge_a1.0", "answer", RidgeClassifier(alpha=1.0, class_weight="balanced")),
        Candidate("answer_ridge_a3.0", "answer", RidgeClassifier(alpha=3.0, class_weight="balanced")),
        Candidate(
            "answer_sgd_log",
            "answer",
            SGDClassifier(
                loss="log_loss",
                penalty="elasticnet",
                alpha=1e-5,
                l1_ratio=0.15,
                class_weight="balanced",
                max_iter=1200,
                tol=1e-3,
                random_state=seed,
            ),
        ),
    ]


def select_text(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column == "combined":
        return (
            "Question: "
            + frame["question"].fillna("").astype(str)
            + "\nAnswer: "
            + frame["interview_answer"].fillna("").astype(str)
        ).to_numpy()
    return frame[column].fillna("").astype(str).to_numpy()


def vectorizer_specs(profile: str) -> list[tuple[str, str, TfidfVectorizer]]:
    if profile == "rich":
        return [
            (
                "combined_word",
                "combined",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                    max_features=45_000,
                ),
            ),
            (
                "answer_word",
                "interview_answer",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                    max_features=55_000,
                ),
            ),
            (
                "char",
                "combined",
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
    if profile == "answer":
        return [
            (
                "answer_word",
                "interview_answer",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 3),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                    max_features=80_000,
                ),
            ),
            (
                "answer_char",
                "interview_answer",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="char_wb",
                    ngram_range=(4, 7),
                    min_df=2,
                    sublinear_tf=True,
                    max_features=70_000,
                ),
            ),
        ]
    raise ValueError(f"Unknown profile: {profile}")


def fit_profile(profile: str, train_frame: pd.DataFrame) -> tuple[FeaturePack, object]:
    specs = vectorizer_specs(profile)
    parts = []
    fitted_specs = []
    for name, column, vectorizer in specs:
        matrix = vectorizer.fit_transform(select_text(train_frame, column))
        fitted_specs.append((name, column, vectorizer))
        parts.append(matrix)
    hand = HandcraftedFeatures()
    hand_matrix = hand.transform(train_frame)
    scaler = StandardScaler()
    hand_scaled = scaler.fit_transform(hand_matrix)
    parts.append(csr_matrix(hand_scaled))
    pack = FeaturePack(profile=profile, vectorizers=fitted_specs, hand=hand, scaler=scaler)
    return pack, hstack(parts, format="csr")


def scores_for_estimator(estimator, matrix) -> np.ndarray:
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(matrix)
    elif hasattr(estimator, "predict_log_proba"):
        scores = estimator.predict_log_proba(matrix)
    else:
        scores = np.log(np.clip(estimator.predict_proba(matrix), 1e-12, 1.0))
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    return np.asarray(scores, dtype=np.float64)


def tune_bias(probs: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, float]:
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    best_bias = np.zeros(len(LABELS), dtype=np.float64)
    best_score = -1.0
    grid = np.round(np.arange(-3.0, 3.01, 0.05), 2)
    for amb_bias in grid:
        for nr_bias in grid:
            bias = np.array([0.0, amb_bias, nr_bias])
            pred = (logits + bias).argmax(axis=1)
            score = f1_score(y_true, pred, average="macro")
            if score > best_score:
                best_score = float(score)
                best_bias = bias
    return best_bias, best_score


def evaluate_probs(frame: pd.DataFrame, probs: np.ndarray, bias: np.ndarray, output_dir: Path, split: str) -> dict:
    logits = np.log(np.clip(probs, 1e-12, 1.0)) + bias
    preds = logits.argmax(axis=1)
    y_true = frame["label_id"].to_numpy()
    metrics = classification_metrics(y_true, preds, LABELS)
    write_json(metrics, output_dir / f"{split}_metrics.json")
    write_report(metrics, LABELS, output_dir / f"{split}_classification_report.csv")
    out = frame[["question", "interview_answer", LABEL_COLUMN]].copy()
    out["prediction"] = [LABELS[int(i)] for i in preds]
    for idx, label in enumerate(LABELS):
        out[f"prob_{label}"] = probs[:, idx]
    out.to_csv(output_dir / f"{split}_predictions.csv", index=False)
    plot_confusion_matrix(metrics["confusion_matrix"], LABELS, output_dir / f"{split}_confusion.png")
    return metrics


def greedy_select(candidate_probs: dict[str, np.ndarray], y_true: np.ndarray, max_selected: int):
    selected: list[str] = []
    remaining = set(candidate_probs)
    best_score = -1.0
    best_bias = np.zeros(len(LABELS), dtype=np.float64)
    while remaining and len(selected) < max_selected:
        step_best = None
        for name in sorted(remaining):
            names = selected + [name]
            probs = np.mean([candidate_probs[n] for n in names], axis=0)
            bias, score = tune_bias(probs, y_true)
            if step_best is None or score > step_best[0]:
                step_best = (score, name, bias)
        if step_best is None or step_best[0] <= best_score + 1e-9:
            break
        best_score, name, best_bias = step_best
        selected.append(name)
        remaining.remove(name)
    return selected, best_bias, best_score


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train, final = load_qevasion()
    y = train["label_id"].to_numpy()
    groups = train["group_id"].to_numpy()
    candidate_list = candidates(args.seed)
    candidate_by_name = {candidate.name: candidate for candidate in candidate_list}
    oof_probs = {candidate.name: np.zeros((len(train), len(LABELS)), dtype=np.float64) for candidate in candidate_list}
    fold_rows = []

    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold, (train_idx, dev_idx) in enumerate(splitter.split(train, y, groups), start=1):
        train_fold = train.iloc[train_idx]
        dev_fold = train.iloc[dev_idx]
        y_train = y[train_idx]
        y_dev = y[dev_idx]
        profile_cache = {}
        for profile in sorted({candidate.profile for candidate in candidate_list}):
            pack, x_train = fit_profile(profile, train_fold)
            x_dev = pack.transform(dev_fold)
            profile_cache[profile] = (x_train, x_dev)
        for candidate in candidate_list:
            x_train, x_dev = profile_cache[candidate.profile]
            estimator = clone(candidate.estimator)
            estimator.fit(x_train, y_train)
            probs = softmax(scores_for_estimator(estimator, x_dev), axis=1)
            oof_probs[candidate.name][dev_idx] = probs
            raw_score = f1_score(y_dev, probs.argmax(axis=1), average="macro")
            fold_rows.append({"fold": fold, "candidate": candidate.name, "raw_macro_f1": float(raw_score)})
            print(f"fold={fold} candidate={candidate.name} raw_macro_f1={raw_score:.6f}")

    candidate_rows = []
    for candidate in candidate_list:
        bias, score = tune_bias(oof_probs[candidate.name], y)
        candidate_rows.append({"candidate": candidate.name, "oof_macro_f1": score, "bias": bias.tolist()})
    pd.DataFrame(candidate_rows).sort_values("oof_macro_f1", ascending=False).to_csv(
        args.output_dir / "candidate_oof_scores.csv", index=False
    )
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "fold_scores.csv", index=False)

    selected, bias, selected_score = greedy_select(oof_probs, y, args.max_selected)
    selected_oof = np.mean([oof_probs[name] for name in selected], axis=0)
    oof_metrics = evaluate_probs(train, selected_oof, bias, args.output_dir, "oof")
    print(f"selected={selected} oof_macro_f1={oof_metrics['macro_f1']:.6f}")

    final_models = []
    final_probs_by_name = {}
    for profile in sorted({candidate_by_name[name].profile for name in selected}):
        pack, x_train = fit_profile(profile, train)
        x_final = pack.transform(final) if args.eval_final else None
        for name in selected:
            candidate = candidate_by_name[name]
            if candidate.profile != profile:
                continue
            estimator = clone(candidate.estimator)
            estimator.fit(x_train, y)
            final_models.append({"name": name, "profile": profile, "pack": pack, "estimator": estimator})
            if args.eval_final and x_final is not None:
                final_probs_by_name[name] = softmax(scores_for_estimator(estimator, x_final), axis=1)
    joblib.dump(final_models, args.output_dir / "selected_models.joblib")
    metadata = {
        "seed": args.seed,
        "folds": args.folds,
        "selected": selected,
        "bias": bias.tolist(),
        "oof_macro_f1": oof_metrics["macro_f1"],
        "train_rows": len(train),
        "final_rows": len(final),
        "candidate_scores": candidate_rows,
        "uses_only_text_fields_for_modeling": ["question", "interview_answer"],
    }
    write_json(metadata, args.output_dir / "metadata.json")

    if args.eval_final:
        final_probs = np.mean([final_probs_by_name[name] for name in selected], axis=0)
        final_metrics = evaluate_probs(final, final_probs, bias, args.output_dir, "final")
        print(f"final_macro_f1={final_metrics['macro_f1']:.6f}")


if __name__ == "__main__":
    main()
