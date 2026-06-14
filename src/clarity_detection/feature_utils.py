from __future__ import annotations

import re

import numpy as np
import pandas as pd


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']+")

DIRECT_MARKERS = (
    "yes",
    "no",
    "absolutely",
    "correct",
    "incorrect",
    "true",
    "false",
    "i agree",
    "i disagree",
)

EVASION_MARKERS = (
    "look",
    "well",
    "let me",
    "what i",
    "the fact is",
    "as i said",
    "i think",
    "i believe",
    "we need",
    "we have to",
    "going forward",
    "at the end of the day",
    "that's not",
    "i'm not going to",
    "i don't want to",
)


def words(text: str | None) -> list[str]:
    return [match.group(0).lower() for match in _WORD_RE.finditer(str(text or ""))]


def count_markers(text: str | None, markers: tuple[str, ...]) -> int:
    lowered = f" {str(text or '').lower()} "
    return sum(lowered.count(marker) for marker in markers)


def numeric_text_features(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for question, answer in zip(frame["question"].tolist(), frame["interview_answer"].tolist()):
        q_words = words(question)
        a_words = words(answer)
        q_set = set(q_words)
        a_set = set(a_words)
        overlap = len(q_set & a_set)
        answer_text = str(answer or "")
        first_tokens = a_words[:8]
        first_text = " ".join(first_tokens)
        rows.append(
            {
                "question_words": len(q_words),
                "answer_words": len(a_words),
                "answer_chars": len(answer_text),
                "answer_sentences": max(1, answer_text.count(".") + answer_text.count("?") + answer_text.count("!")),
                "question_answer_overlap": overlap / max(1, len(q_set)),
                "answer_question_overlap": overlap / max(1, len(a_set)),
                "answer_to_question_len": len(a_words) / max(1, len(q_words)),
                "direct_marker_count": count_markers(answer, DIRECT_MARKERS),
                "evasion_marker_count": count_markers(answer, EVASION_MARKERS),
                "starts_with_direct_marker": float(any(first_text.startswith(marker) for marker in DIRECT_MARKERS)),
                "starts_with_evasion_marker": float(any(first_text.startswith(marker) for marker in EVASION_MARKERS)),
                "has_question_back": float("?" in answer_text),
            }
        )
    return pd.DataFrame(rows)


def probability_features(frame: pd.DataFrame) -> pd.DataFrame:
    prob_cols = [col for col in frame.columns if col.startswith("prob_")]
    probs = frame[prob_cols].astype(float).to_numpy()
    probs = np.clip(probs, 1e-8, 1.0)
    sorted_probs = np.sort(probs, axis=1)
    entropy = -(probs * np.log(probs)).sum(axis=1)
    features = {f"log_{col}": np.log(probs[:, idx]) for idx, col in enumerate(prob_cols)}
    features.update(
        {
            "prob_margin_top2": sorted_probs[:, -1] - sorted_probs[:, -2],
            "prob_entropy": entropy,
            "prob_max": sorted_probs[:, -1],
        }
    )
    return pd.DataFrame(features)


def build_decision_features(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [
            probability_features(frame).reset_index(drop=True),
            numeric_text_features(frame).reset_index(drop=True),
        ],
        axis=1,
    )
