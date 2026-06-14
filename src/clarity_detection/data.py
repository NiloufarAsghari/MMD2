from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import StratifiedGroupKFold

from .labels import (
    BOUNDARY_LABELS,
    CLEAR_BOUNDARY_LABEL2ID,
    LABEL2ID,
    REPLY_BOUNDARY_LABEL2ID,
    REPLY_BOUNDARY_LABELS,
    normalize_label,
)


DATASET_NAME = "ailsntua/QEvasion"
TEXT_COLUMNS = ["question", "interview_answer"]
LABEL_COLUMN = "clarity_label"


@dataclass(frozen=True)
class SplitBundle:
    train: pd.DataFrame
    dev: pd.DataFrame
    final: pd.DataFrame


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


GROUP_MODES = ["url_question_answer", "qa_text", "answer_text", "question_text"]


def stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()


def stable_group_id(row: pd.Series, mode: str = "url_question_answer") -> str:
    if mode == "url_question_answer":
        raw = "|".join(
            [
                str(row.get("url", "")),
                str(row.get("question_order", "")),
                str(row.get("interview_answer", "")),
            ]
        )
    elif mode == "qa_text":
        raw = "|".join([str(row.get("question", "")), str(row.get("interview_answer", ""))])
    elif mode == "answer_text":
        raw = str(row.get("interview_answer", ""))
    elif mode == "question_text":
        raw = str(row.get("question", ""))
    else:
        raise ValueError(f"Unknown group mode: {mode}")
    return stable_hash(raw)


def add_group_ids(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode not in GROUP_MODES:
        raise ValueError(f"Unknown group mode: {mode}")
    frame = frame.copy()
    frame["group_id"] = frame.apply(lambda row: stable_group_id(row, mode), axis=1)
    frame["group_mode"] = mode
    return frame


def stable_text_key(*parts: str) -> str:
    raw = "\n".join(str(part or "").lower().strip() for part in parts)
    return stable_hash(raw)


def answer_key(frame: pd.DataFrame) -> pd.Series:
    return frame["interview_answer"].map(lambda value: stable_text_key(value))


def qa_key(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [stable_text_key(q, a) for q, a in zip(frame["question"], frame["interview_answer"])],
        index=frame.index,
    )


def stable_group_id_legacy(row: pd.Series) -> str:
    raw = "|".join(
        [
            str(row.get("url", "")),
            str(row.get("question_order", "")),
            str(row.get("interview_answer", "")),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()


def format_pair(question: str | None, answer: str | None) -> str:
    question = clean_text(question)
    answer = clean_text(answer)
    return f"Question: {question}\nAnswer: {answer}"


def clean_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split())


def load_qevasion() -> tuple[pd.DataFrame, pd.DataFrame]:
    ds = load_dataset(DATASET_NAME)
    train = ds["train"].to_pandas()
    final = ds["test"].to_pandas()
    for frame in (train, final):
        frame[LABEL_COLUMN] = frame[LABEL_COLUMN].map(normalize_label)
        frame["question"] = frame["question"].map(clean_text)
        frame["interview_answer"] = frame["interview_answer"].map(clean_text)
        frame["text"] = [
            format_pair(q, a)
            for q, a in zip(frame["question"].tolist(), frame["interview_answer"].tolist())
        ]
        frame["group_id"] = frame.apply(stable_group_id, axis=1)
        frame["group_mode"] = "url_question_answer"
        frame["label_id"] = frame[LABEL_COLUMN].map(LABEL2ID).astype(int)
    return train, final


def make_internal_split(
    train_df: pd.DataFrame,
    *,
    seed: int = 13,
    dev_folds: int = 5,
    group_mode: str = "url_question_answer",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = add_group_ids(train_df, group_mode)
    splitter = StratifiedGroupKFold(n_splits=dev_folds, shuffle=True, random_state=seed)
    y = train_df["label_id"].to_numpy()
    groups = train_df["group_id"].to_numpy()
    train_idx, dev_idx = next(splitter.split(train_df, y, groups))
    train_split = train_df.iloc[train_idx].reset_index(drop=True)
    dev_split = train_df.iloc[dev_idx].reset_index(drop=True)
    return train_split, dev_split


def get_splits(
    *,
    seed: int = 13,
    dev_folds: int = 5,
    task: str = "multiclass",
    group_mode: str = "url_question_answer",
    max_train_samples: int | None = None,
    max_dev_samples: int | None = None,
    train_on_full: bool = False,
) -> SplitBundle:
    set_seed(seed)
    official_train, final = load_qevasion()
    train, dev = make_internal_split(official_train, seed=seed, dev_folds=dev_folds, group_mode=group_mode)
    if train_on_full:
        train = official_train.reset_index(drop=True)

    if task == "boundary":
        train = train[train[LABEL_COLUMN].isin(BOUNDARY_LABELS)].reset_index(drop=True)
        dev = dev[dev[LABEL_COLUMN].isin(BOUNDARY_LABELS)].reset_index(drop=True)
        final = final[final[LABEL_COLUMN].isin(BOUNDARY_LABELS)].reset_index(drop=True)
    elif task == "reply_boundary":
        train = train[train[LABEL_COLUMN].isin(REPLY_BOUNDARY_LABELS)].reset_index(drop=True)
        dev = dev[dev[LABEL_COLUMN].isin(REPLY_BOUNDARY_LABELS)].reset_index(drop=True)
        final = final[final[LABEL_COLUMN].isin(REPLY_BOUNDARY_LABELS)].reset_index(drop=True)
        for frame in (train, dev, final):
            frame["label_id"] = frame[LABEL_COLUMN].map(REPLY_BOUNDARY_LABEL2ID).astype(int)
    elif task == "clear_boundary":
        for frame in (train, dev, final):
            frame[LABEL_COLUMN] = np.where(frame[LABEL_COLUMN].eq("Clear Reply"), "Clear Reply", "Non-Clear")
            frame["label_id"] = frame[LABEL_COLUMN].map(CLEAR_BOUNDARY_LABEL2ID).astype(int)
    elif task != "multiclass":
        raise ValueError(f"Unknown task: {task}")

    if max_train_samples is not None:
        train = stratified_sample(train, max_train_samples, seed=seed)
    if max_dev_samples is not None:
        dev = stratified_sample(dev, max_dev_samples, seed=seed)

    return SplitBundle(train=train, dev=dev, final=final.reset_index(drop=True))


def stratified_sample(df: pd.DataFrame, n: int, *, seed: int) -> pd.DataFrame:
    if n >= len(df):
        return df.reset_index(drop=True)
    parts = []
    rng = np.random.default_rng(seed)
    counts = df[LABEL_COLUMN].value_counts(normalize=True)
    remaining = n
    labels = list(counts.index)
    for label in labels[:-1]:
        take = max(1, int(round(counts[label] * n)))
        subset = df[df[LABEL_COLUMN] == label]
        take = min(take, len(subset), remaining - (len(labels) - len(parts) - 1))
        parts.append(subset.sample(n=take, random_state=int(rng.integers(0, 1_000_000))))
        remaining -= take
    last_subset = df[df[LABEL_COLUMN] == labels[-1]]
    parts.append(last_subset.sample(n=min(remaining, len(last_subset)), random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def save_split_summary(bundle: SplitBundle, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, frame in [("train", bundle.train), ("dev", bundle.dev), ("final", bundle.final)]:
        for label, count in frame[LABEL_COLUMN].value_counts().sort_index().items():
            rows.append({"split": name, "label": label, "count": int(count)})
    pd.DataFrame(rows).to_csv(output_dir / "split_label_counts.csv", index=False)
