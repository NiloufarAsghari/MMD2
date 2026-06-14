from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import LABEL_COLUMN, get_splits
from .labels import LABELS
from .metrics import classification_metrics, write_json, write_report
from .plots import plot_confusion_matrix


HYPOTHESIS_SETS = {
    "directness": {
        "Clear Reply": "The answer clearly and directly answers the question.",
        "Ambivalent": "The answer partially answers the question, hedges, or is ambiguous.",
        "Clear Non-Reply": "The answer refuses, redirects, changes the subject, or does not answer the question.",
    },
    "rubric": {
        "Clear Reply": "The answer gives a clear reply to the specific question asked.",
        "Ambivalent": "The answer is unclear, partial, mixed, or ambivalent with respect to the question.",
        "Clear Non-Reply": "The answer is a clear non-reply to the question.",
    },
}


class NLIPairDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer,
        *,
        hypotheses: dict[str, str],
        max_length: int,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.hypotheses = hypotheses
        self.max_length = max_length
        self.items = [(row_idx, label) for row_idx in range(len(self.frame)) for label in LABELS]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        row_idx, label = self.items[idx]
        row = self.frame.iloc[row_idx]
        premise = "Question: " + str(row["question"]) + "\nAnswer: " + str(row["interview_answer"])
        hypothesis = self.hypotheses[label]
        enc = self.tokenizer(
            premise,
            hypothesis,
            truncation="only_first",
            max_length=self.max_length,
            padding=False,
        )
        enc["row_idx"] = row_idx
        enc["label_idx"] = LABELS.index(label)
        return enc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run zero-shot NLI clarity classification.")
    parser.add_argument("--model-name", default="MoritzLaurer/deberta-v3-base-zeroshot-v2.0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--split", choices=["dev", "final", "both"], default="dev")
    parser.add_argument("--hypothesis-set", choices=sorted(HYPOTHESIS_SETS), default="directness")
    parser.add_argument("--score-mode", choices=["entailment", "entailment_minus_contradiction"], default="entailment")
    parser.add_argument("--calibrate-dev-prior", action="store_true")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max-dev-samples", type=int)
    parser.add_argument("--max-final-samples", type=int)
    return parser.parse_args()


def collate(tokenizer, batch: list[dict]) -> dict[str, torch.Tensor]:
    row_idx = torch.tensor([item.pop("row_idx") for item in batch], dtype=torch.long)
    label_idx = torch.tensor([item.pop("label_idx") for item in batch], dtype=torch.long)
    enc = tokenizer.pad(batch, padding=True, return_tensors="pt")
    enc["row_idx"] = row_idx
    enc["label_idx"] = label_idx
    return enc


def label_index(config, name: str) -> int | None:
    for idx, label in config.id2label.items():
        if name.lower() in str(label).lower():
            return int(idx)
    return None


def score_logits(logits: np.ndarray, *, entail_idx: int, contradiction_idx: int | None, mode: str) -> np.ndarray:
    if mode == "entailment_minus_contradiction" and contradiction_idx is not None:
        return logits[:, entail_idx] - logits[:, contradiction_idx]
    return logits[:, entail_idx]


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / exp.sum(axis=1, keepdims=True)


def predict_split(
    frame: pd.DataFrame,
    *,
    tokenizer,
    model,
    hypotheses: dict[str, str],
    device: torch.device,
    args: argparse.Namespace,
) -> np.ndarray:
    ds = NLIPairDataset(frame, tokenizer, hypotheses=hypotheses, max_length=args.max_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda batch: collate(tokenizer, batch))
    entail_idx = label_index(model.config, "entail")
    contradiction_idx = label_index(model.config, "contrad")
    if entail_idx is None:
        raise ValueError(f"Could not find entailment label in id2label={model.config.id2label}")

    scores = np.zeros((len(frame), len(LABELS)), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="zero-shot nli", leave=False):
            row_idx = batch.pop("row_idx").numpy()
            label_idx = batch.pop("label_idx").numpy()
            batch = {key: value.to(device) for key, value in batch.items()}
            ctx = torch.amp.autocast("cuda", dtype=torch.float16) if args.fp16 and device.type == "cuda" else nullcontext()
            with ctx:
                outputs = model(**batch)
            batch_scores = score_logits(
                outputs.logits.detach().float().cpu().numpy(),
                entail_idx=entail_idx,
                contradiction_idx=contradiction_idx,
                mode=args.score_mode,
            )
            scores[row_idx, label_idx] = batch_scores
    return softmax(scores)


def write_predictions(frame: pd.DataFrame, probs: np.ndarray, output_dir: Path, split: str) -> dict:
    preds = probs.argmax(axis=1)
    y_true = frame["label_id"].to_numpy()
    metrics = classification_metrics(y_true, preds, LABELS)
    pred_frame = frame[["question", "interview_answer", LABEL_COLUMN]].copy()
    pred_frame["prediction"] = [LABELS[idx] for idx in preds]
    for idx, label in enumerate(LABELS):
        pred_frame[f"prob_{label}"] = probs[:, idx]
    pred_frame.to_csv(output_dir / f"{split}_predictions.csv", index=False)
    write_json(metrics, output_dir / f"{split}_metrics.json")
    write_report(metrics, LABELS, output_dir / f"{split}_classification_report.csv")
    plot_confusion_matrix(metrics["confusion_matrix"], LABELS, output_dir / f"{split}_confusion.png")
    return metrics


def tune_bias(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    best_bias = np.zeros(len(LABELS), dtype=np.float64)
    best_f1 = -1.0
    grid = np.round(np.arange(-4.0, 4.01, 0.1), 2)
    for amb_bias in grid:
        for nr_bias in grid:
            bias = np.array([0.0, amb_bias, nr_bias])
            preds = (logits + bias).argmax(axis=1)
            score = f1_score(y_true, preds, average="macro")
            if score > best_f1:
                best_f1 = score
                best_bias = bias
    return best_bias


def apply_bias(probs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-12, 1.0)) + bias
    return softmax(logits)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = get_splits(seed=args.seed, max_dev_samples=args.max_dev_samples)
    if args.max_final_samples is not None:
        splits = type(splits)(
            train=splits.train,
            dev=splits.dev,
            final=splits.final.sample(n=min(args.max_final_samples, len(splits.final)), random_state=args.seed).reset_index(drop=True),
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    hypotheses = HYPOTHESIS_SETS[args.hypothesis_set]
    write_json(
        {
            "model_name": args.model_name,
            "hypothesis_set": args.hypothesis_set,
            "hypotheses": hypotheses,
            "score_mode": args.score_mode,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "fp16": args.fp16,
            "calibrate_dev_prior": args.calibrate_dev_prior,
            "id2label": {str(k): str(v) for k, v in model.config.id2label.items()},
        },
        args.output_dir / "zero_shot_config.json",
    )

    metrics = {}
    bias = np.zeros(len(LABELS), dtype=np.float64)
    if args.split in {"dev", "both"}:
        dev_probs = predict_split(splits.dev, tokenizer=tokenizer, model=model, hypotheses=hypotheses, device=device, args=args)
        if args.calibrate_dev_prior:
            bias = tune_bias(dev_probs, splits.dev["label_id"].to_numpy())
            write_json({"bias": bias.tolist(), "labels": LABELS}, args.output_dir / "calibration.json")
            dev_probs = apply_bias(dev_probs, bias)
        metrics["dev"] = write_predictions(splits.dev, dev_probs, args.output_dir, "dev")
        print(f"dev_macro_f1={metrics['dev']['macro_f1']:.6f}")
    if args.split in {"final", "both"}:
        final_probs = predict_split(splits.final, tokenizer=tokenizer, model=model, hypotheses=hypotheses, device=device, args=args)
        if args.calibrate_dev_prior:
            final_probs = apply_bias(final_probs, bias)
        metrics["final"] = write_predictions(splits.final, final_probs, args.output_dir, "final")
        print(f"final_macro_f1={metrics['final']['macro_f1']:.6f}")
    write_json(metrics, args.output_dir / "zero_shot_summary.json")


if __name__ == "__main__":
    main()
