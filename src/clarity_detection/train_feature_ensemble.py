from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .data import LABEL_COLUMN, load_qevasion
from .labels import LABELS
from .metrics import classification_metrics, write_json, write_report
from .plots import plot_confusion_matrix


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "this",
    "to",
    "we",
    "what",
    "will",
    "with",
    "you",
    "your",
}

YES_NO_STARTS = {
    "am",
    "are",
    "can",
    "could",
    "did",
    "do",
    "does",
    "has",
    "have",
    "is",
    "should",
    "was",
    "were",
    "will",
    "would",
}

QUESTION_STARTS = [
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "is",
    "are",
    "do",
    "did",
    "does",
    "can",
    "could",
    "will",
    "would",
    "should",
]

DIRECT_CUES = [
    "yes",
    "no",
    "absolutely",
    "i agree",
    "i do",
    "i don't",
    "we will",
    "we won't",
    "i will",
    "i won't",
    "guarantee",
    "there is no question",
    "that's right",
    "that's wrong",
]

EVASION_CUES = [
    "look",
    "listen",
    "well",
    "let me",
    "you know",
    "i think",
    "i believe",
    "going to",
    "we're going to",
    "i can't",
    "i cannot",
    "not going to",
    "i'm not going to",
    "we'll see",
    "depends",
    "that's a hypothetical",
    "i don't want to",
    "i'm not sure",
    "as i said",
    "as i've said",
]

CONTRAST_CUES = [
    "but",
    "however",
    "although",
    "though",
    "nevertheless",
    "instead",
    "on the other hand",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grouped-CV engineered feature ensemble.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/feature_ensemble_cv"))
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-selected", type=int, default=4)
    parser.add_argument("--include-logreg", action="store_true")
    parser.add_argument("--eval-final", action="store_true")
    return parser.parse_args()


class TextColumn(BaseEstimator, TransformerMixin):
    def __init__(self, column: str):
        self.column = column

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.column].fillna("").astype(str).to_numpy()


class CombinedText(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        question = X["question"].fillna("").astype(str)
        answer = X["interview_answer"].fillna("").astype(str)
        return ("Question: " + question + "\nAnswer: " + answer).to_numpy()


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z']+", str(text).lower())


def content_tokens(text: str) -> set[str]:
    return {tok for tok in word_tokens(text) if tok not in STOPWORDS and len(tok) > 2}


def count_contains(text: str, cues: list[str]) -> int:
    lower = str(text).lower()
    return sum(1 for cue in cues if cue in lower)


class HandcraftedFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        rows: list[list[float]] = []
        for _, row in X.iterrows():
            question = str(row.get("question", ""))
            answer = str(row.get("interview_answer", ""))
            q_lower = question.lower()
            a_lower = answer.lower()
            q_words = word_tokens(question)
            a_words = word_tokens(answer)
            q_content = content_tokens(question)
            a_content = content_tokens(answer)
            first_answer = " ".join(a_words[:40])
            last_answer = " ".join(a_words[-40:])
            overlap = q_content & a_content
            first_overlap = q_content & set(first_answer.split())
            last_overlap = q_content & set(last_answer.split())
            q_first = q_words[0] if q_words else ""
            yes_no_question = float(q_first in YES_NO_STARTS)
            sentence_count = max(1, len(re.findall(r"[.!?]+", answer)))
            answer_sentence_avg = len(a_words) / sentence_count
            numeric_count = len(re.findall(r"\b\d+[\d,.%$-]*\b", answer))
            rows.append(
                [
                    math.log1p(len(question)),
                    math.log1p(len(answer)),
                    math.log1p(len(q_words)),
                    math.log1p(len(a_words)),
                    math.log1p(len(a_words) / max(1, len(q_words))),
                    math.log1p(sentence_count),
                    math.log1p(answer_sentence_avg),
                    math.log1p(numeric_count),
                    len(overlap) / max(1, len(q_content)),
                    len(overlap) / max(1, len(a_content)),
                    len(first_overlap) / max(1, len(q_content)),
                    len(last_overlap) / max(1, len(q_content)),
                    yes_no_question,
                    float(a_words[:1] in [["yes"], ["no"]]),
                    float(any(tok in {"yes", "no"} for tok in a_words[:12])),
                    count_contains(a_lower, DIRECT_CUES),
                    count_contains(a_lower, EVASION_CUES),
                    count_contains(a_lower, CONTRAST_CUES),
                    a_lower.count("?"),
                    a_lower.count("!"),
                    float("not going to" in a_lower or "i'm not going to" in a_lower),
                    float("i don't know" in a_lower or "i do not know" in a_lower),
                    float("i can't" in a_lower or "i cannot" in a_lower),
                    float("guarantee" in a_lower or "absolutely" in a_lower),
                    float("hypothetical" in a_lower),
                    float("?" in question),
                    *[float(q_lower.startswith(start + " ")) for start in QUESTION_STARTS],
                ]
            )
        return np.asarray(rows, dtype=np.float32)


def make_feature_union(profile: str) -> FeatureUnion:
    if profile == "rich":
        return FeatureUnion(
            [
                (
                    "combined_word",
                    Pipeline(
                        [
                            ("text", CombinedText()),
                            (
                                "tfidf",
                                TfidfVectorizer(
                                    lowercase=True,
                                    strip_accents="unicode",
                                    ngram_range=(1, 2),
                                    min_df=2,
                                    max_df=0.98,
                                    sublinear_tf=True,
                    max_features=40_000,
                                ),
                            ),
                        ]
                    ),
                ),
                (
                    "answer_word",
                    Pipeline(
                        [
                            ("text", TextColumn("interview_answer")),
                            (
                                "tfidf",
                                TfidfVectorizer(
                                    lowercase=True,
                                    strip_accents="unicode",
                                    ngram_range=(1, 3),
                                    min_df=2,
                                    max_df=0.98,
                                    sublinear_tf=True,
                                    max_features=50_000,
                                ),
                            ),
                        ]
                    ),
                ),
                (
                    "question_word",
                    Pipeline(
                        [
                            ("text", TextColumn("question")),
                            (
                                "tfidf",
                                TfidfVectorizer(
                                    lowercase=True,
                                    strip_accents="unicode",
                                    ngram_range=(1, 2),
                                    min_df=1,
                                    sublinear_tf=True,
                                    max_features=12_000,
                                ),
                            ),
                        ]
                    ),
                ),
                (
                    "char",
                    Pipeline(
                        [
                            ("text", CombinedText()),
                            (
                                "tfidf",
                                TfidfVectorizer(
                                    lowercase=True,
                                    analyzer="char_wb",
                                    ngram_range=(3, 6),
                                    min_df=2,
                                    sublinear_tf=True,
                                    max_features=50_000,
                                ),
                            ),
                        ]
                    ),
                ),
                ("hand", Pipeline([("features", HandcraftedFeatures()), ("scale", StandardScaler())])),
            ]
        )
    if profile == "answer_heavy":
        return FeatureUnion(
            [
                (
                    "answer_word",
                    Pipeline(
                        [
                            ("text", TextColumn("interview_answer")),
                            (
                                "tfidf",
                                TfidfVectorizer(
                                    lowercase=True,
                                    strip_accents="unicode",
                                    ngram_range=(1, 3),
                                    min_df=2,
                                    max_df=0.98,
                                    sublinear_tf=True,
                                    max_features=70_000,
                                ),
                            ),
                        ]
                    ),
                ),
                (
                    "answer_char",
                    Pipeline(
                        [
                            ("text", TextColumn("interview_answer")),
                            (
                                "tfidf",
                                TfidfVectorizer(
                                    lowercase=True,
                                    analyzer="char_wb",
                                    ngram_range=(4, 7),
                                    min_df=2,
                                    sublinear_tf=True,
                                    max_features=50_000,
                                ),
                            ),
                        ]
                    ),
                ),
                ("hand", Pipeline([("features", HandcraftedFeatures()), ("scale", StandardScaler())])),
            ]
        )
    raise ValueError(f"Unknown profile: {profile}")


@dataclass(frozen=True)
class Candidate:
    name: str
    profile: str
    estimator: object


def candidate_grid(seed: int, *, include_logreg: bool = False) -> list[Candidate]:
    candidates = [
        Candidate("rich_svc_c0.5", "rich", LinearSVC(C=0.5, class_weight="balanced", max_iter=6000, random_state=seed)),
        Candidate("rich_svc_c1.0", "rich", LinearSVC(C=1.0, class_weight="balanced", max_iter=6000, random_state=seed)),
        Candidate("rich_ridge_a1.0", "rich", RidgeClassifier(alpha=1.0, class_weight="balanced")),
        Candidate(
            "answer_svc_c1.0",
            "answer_heavy",
            LinearSVC(C=1.0, class_weight="balanced", max_iter=6000, random_state=seed),
        ),
        Candidate("answer_ridge_a1.0", "answer_heavy", RidgeClassifier(alpha=1.0, class_weight="balanced")),
    ]
    if include_logreg:
        candidates.extend(
            [
                Candidate(
                    "rich_logreg_c0.7",
                    "rich",
                    LogisticRegression(C=0.7, class_weight="balanced", max_iter=500, solver="liblinear", random_state=seed),
                ),
                Candidate(
                    "answer_logreg_c0.7",
                    "answer_heavy",
                    LogisticRegression(C=0.7, class_weight="balanced", max_iter=500, solver="liblinear", random_state=seed),
                ),
            ]
        )
    return candidates


def make_pipeline(candidate: Candidate) -> Pipeline:
    return Pipeline([("features", make_feature_union(candidate.profile)), ("clf", clone(candidate.estimator))])


def scores_for_model(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    clf = model.named_steps["clf"]
    if hasattr(clf, "predict_log_proba"):
        scores = clf.predict_log_proba(model.named_steps["features"].transform(frame))
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(frame)
    else:
        scores = model.predict_proba(frame)
        scores = np.log(np.clip(scores, 1e-12, 1.0))
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    return np.asarray(scores, dtype=np.float64)


def probs_from_scores(scores: np.ndarray) -> np.ndarray:
    return softmax(scores, axis=1)


def tune_bias(probs: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, float]:
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    best_bias = np.zeros(probs.shape[1], dtype=np.float64)
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


def write_predictions(frame: pd.DataFrame, probs: np.ndarray, preds: np.ndarray, path: Path) -> None:
    out = frame[["question", "interview_answer", LABEL_COLUMN]].copy()
    out["prediction"] = [LABELS[int(i)] for i in preds]
    for idx, label in enumerate(LABELS):
        out[f"prob_{label}"] = probs[:, idx]
    out.to_csv(path, index=False)


def evaluate_probs(frame: pd.DataFrame, probs: np.ndarray, bias: np.ndarray, output_dir: Path, split: str) -> dict:
    logits = np.log(np.clip(probs, 1e-12, 1.0)) + bias
    preds = logits.argmax(axis=1)
    y_true = frame["label_id"].to_numpy()
    metrics = classification_metrics(y_true, preds, LABELS)
    write_json(metrics, output_dir / f"{split}_metrics.json")
    write_report(metrics, LABELS, output_dir / f"{split}_classification_report.csv")
    write_predictions(frame, probs, preds, output_dir / f"{split}_predictions.csv")
    plot_confusion_matrix(metrics["confusion_matrix"], LABELS, output_dir / f"{split}_confusion.png")
    return metrics


def greedy_select(candidate_probs: dict[str, np.ndarray], y_true: np.ndarray, max_selected: int) -> tuple[list[str], np.ndarray, float]:
    remaining = set(candidate_probs)
    selected: list[str] = []
    best_probs: np.ndarray | None = None
    best_bias = np.zeros(len(LABELS), dtype=np.float64)
    best_score = -1.0
    while remaining and len(selected) < max_selected:
        step_best: tuple[float, str, np.ndarray, np.ndarray] | None = None
        for name in sorted(remaining):
            names = selected + [name]
            probs = np.mean([candidate_probs[n] for n in names], axis=0)
            bias, score = tune_bias(probs, y_true)
            if step_best is None or score > step_best[0]:
                step_best = (score, name, probs, bias)
        if step_best is None or step_best[0] <= best_score + 1e-9:
            break
        best_score, best_name, best_probs, best_bias = step_best
        selected.append(best_name)
        remaining.remove(best_name)
    if best_probs is None:
        raise RuntimeError("No candidates selected.")
    return selected, best_bias, best_score


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train, final = load_qevasion()
    y = train["label_id"].to_numpy()
    groups = train["group_id"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    candidates = candidate_grid(args.seed, include_logreg=args.include_logreg)

    oof_probs: dict[str, np.ndarray] = {candidate.name: np.zeros((len(train), len(LABELS)), dtype=np.float64) for candidate in candidates}
    fold_rows = []
    for fold, (train_idx, dev_idx) in enumerate(splitter.split(train, y, groups), start=1):
        train_fold = train.iloc[train_idx]
        dev_fold = train.iloc[dev_idx]
        y_train = y[train_idx]
        for candidate in candidates:
            model = make_pipeline(candidate)
            model.fit(train_fold, y_train)
            probs = probs_from_scores(scores_for_model(model, dev_fold))
            oof_probs[candidate.name][dev_idx] = probs
            fold_pred = probs.argmax(axis=1)
            fold_score = f1_score(y[dev_idx], fold_pred, average="macro")
            fold_rows.append({"fold": fold, "candidate": candidate.name, "raw_macro_f1": float(fold_score)})
            print(f"fold={fold} candidate={candidate.name} raw_macro_f1={fold_score:.6f}")

    candidate_rows = []
    for candidate in candidates:
        bias, score = tune_bias(oof_probs[candidate.name], y)
        candidate_rows.append({"candidate": candidate.name, "oof_macro_f1": score, "bias": bias.tolist()})
    candidate_frame = pd.DataFrame(candidate_rows).sort_values("oof_macro_f1", ascending=False)
    candidate_frame.to_csv(args.output_dir / "candidate_oof_scores.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "fold_scores.csv", index=False)

    selected, bias, oof_score = greedy_select(oof_probs, y, args.max_selected)
    selected_oof_probs = np.mean([oof_probs[name] for name in selected], axis=0)
    oof_metrics = evaluate_probs(train, selected_oof_probs, bias, args.output_dir, "oof")
    print(f"selected={selected} oof_macro_f1={oof_metrics['macro_f1']:.6f}")

    final_models = []
    final_probs_by_name = {}
    for candidate in candidates:
        if candidate.name not in selected:
            continue
        model = make_pipeline(candidate)
        model.fit(train, y)
        final_models.append((candidate.name, model))
        if args.eval_final:
            final_probs_by_name[candidate.name] = probs_from_scores(scores_for_model(model, final))
    joblib.dump(final_models, args.output_dir / "selected_models.joblib")

    metadata = {
        "seed": args.seed,
        "folds": args.folds,
        "labels": LABELS,
        "selected": selected,
        "bias": bias.tolist(),
        "oof_macro_f1": oof_metrics["macro_f1"],
        "candidate_scores": candidate_rows,
        "train_rows": len(train),
        "final_rows": len(final),
        "uses_only_text_fields_for_modeling": ["question", "interview_answer"],
    }
    write_json(metadata, args.output_dir / "metadata.json")

    if args.eval_final:
        final_probs = np.mean([final_probs_by_name[name] for name in selected], axis=0)
        final_metrics = evaluate_probs(final, final_probs, bias, args.output_dir, "final")
        print(f"final_macro_f1={final_metrics['macro_f1']:.6f}")


if __name__ == "__main__":
    main()
