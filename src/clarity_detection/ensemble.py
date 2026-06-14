from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import LABEL_COLUMN, get_splits
from .labels import BOUNDARY_LABELS, CLEAR_BOUNDARY_LABELS, LABEL2ID, LABELS, REPLY_BOUNDARY_LABELS
from .metrics import classification_metrics, write_json, write_report
from .plots import plot_confusion_matrix
from .train_transformer import QPairDataset, collate_batch, move_to_device, softmax, write_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average trained transformer probabilities.")
    parser.add_argument("--model-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--boundary-model-dirs", nargs="*", type=Path, default=[])
    parser.add_argument("--clear-model-dirs", nargs="*", type=Path, default=[])
    parser.add_argument("--reply-model-dirs", nargs="*", type=Path, default=[])
    parser.add_argument("--clear-alpha", type=float, default=1.0)
    parser.add_argument("--reply-alpha", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--eval-final", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bias-json", type=Path)
    parser.add_argument("--max-dev-samples", type=int)
    parser.add_argument("--chunked-inference", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=384)
    parser.add_argument("--chunk-stride", type=int, default=192)
    parser.add_argument("--max-chunks", type=int, default=6)
    parser.add_argument("--chunk-aggregation", choices=["mean", "max", "mean_max", "noisy_or"], default="mean_max")
    return parser.parse_args()


def load_metadata(model_dir: Path) -> dict:
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        metadata_path = model_dir.parent / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_model(model_dir: Path, metadata: dict, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if metadata.get("lora"):
        base = AutoModelForSequenceClassification.from_pretrained(
            metadata["model_name"],
            num_labels=len(metadata["labels"]),
            id2label={i: label for i, label in enumerate(metadata["labels"])},
            label2id={label: i for i, label in enumerate(metadata["labels"])},
        )
        model = PeftModel.from_pretrained(base, model_dir)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model


def predict_model(
    model_dir: Path,
    frame: pd.DataFrame,
    *,
    batch_size: int,
    device: torch.device,
    fp16: bool,
    chunked_inference: bool = False,
    chunk_size: int = 384,
    chunk_stride: int = 192,
    max_chunks: int = 6,
    chunk_aggregation: str = "mean_max",
) -> tuple[np.ndarray, dict]:
    metadata = load_metadata(model_dir)
    tokenizer, model = load_model(model_dir, metadata, device)
    label2id = metadata["label2id"]
    original_index = None
    if chunked_inference:
        frame, original_index = make_chunked_frame(
            frame,
            tokenizer,
            chunk_size=chunk_size,
            chunk_stride=chunk_stride,
            max_chunks=max_chunks,
        )
    ds = QPairDataset(
        frame,
        tokenizer,
        label2id=label2id,
        max_length=int(metadata["max_length"]),
        truncation=metadata["truncation"],
        input_format=metadata.get("input_format", "pair"),
        max_question_tokens=int(metadata.get("max_question_tokens", 96)),
        head_ratio=float(metadata.get("head_ratio", 0.7)),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: collate_batch(tokenizer, batch))
    logits = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"predict {model_dir.name}", leave=False):
            batch = move_to_device(batch, device)
            batch.pop("labels")
            ctx = torch.amp.autocast("cuda", dtype=torch.float16) if fp16 and device.type == "cuda" else nullcontext()
            with ctx:
                outputs = model(**batch)
            logits.append(outputs.logits.detach().float().cpu().numpy())
    probs = softmax(np.concatenate(logits, axis=0))
    if original_index is not None:
        probs = aggregate_chunk_probs(
            probs,
            original_index,
            original_count=int(original_index.max()) + 1,
            mode=chunk_aggregation,
        )
    return probs, metadata


def select_chunk_starts(token_count: int, window: int, stride: int, max_chunks: int) -> list[int]:
    if token_count <= window:
        return [0]
    stride = max(1, stride)
    last_start = max(0, token_count - window)
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    starts = sorted(set(starts))
    if len(starts) <= max_chunks:
        return starts
    if max_chunks <= 2:
        return [0, last_start][:max_chunks]
    middle = np.linspace(1, len(starts) - 2, max_chunks - 2).round().astype(int).tolist()
    selected = [starts[0], *[starts[idx] for idx in middle], starts[-1]]
    return sorted(set(selected))[:max_chunks]


def make_chunked_frame(
    frame: pd.DataFrame,
    tokenizer,
    *,
    chunk_size: int,
    chunk_stride: int,
    max_chunks: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    original_index = []
    for row_idx, row in frame.reset_index(drop=True).iterrows():
        answer_ids = tokenizer(str(row["interview_answer"]), add_special_tokens=False)["input_ids"]
        starts = select_chunk_starts(len(answer_ids), max(1, chunk_size), chunk_stride, max(1, max_chunks))
        for start in starts:
            chunk_ids = answer_ids[start : start + chunk_size]
            chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            new_row = row.copy()
            new_row["interview_answer"] = chunk_text
            rows.append(new_row)
            original_index.append(row_idx)
    return pd.DataFrame(rows).reset_index(drop=True), np.asarray(original_index, dtype=np.int64)


def aggregate_chunk_probs(
    probs: np.ndarray,
    original_index: np.ndarray,
    *,
    original_count: int,
    mode: str,
) -> np.ndarray:
    aggregated = np.zeros((original_count, probs.shape[1]), dtype=np.float64)
    for row_idx in range(original_count):
        chunk_probs = probs[original_index == row_idx]
        if mode == "mean":
            row_probs = chunk_probs.mean(axis=0)
        elif mode == "max":
            row_probs = chunk_probs.max(axis=0)
        elif mode == "noisy_or":
            row_probs = 1.0 - np.prod(1.0 - np.clip(chunk_probs, 0.0, 1.0), axis=0)
        else:
            row_probs = 0.5 * chunk_probs.mean(axis=0) + 0.5 * chunk_probs.max(axis=0)
        aggregated[row_idx] = row_probs / max(row_probs.sum(), 1e-12)
    return aggregated


def apply_boundary(base_probs: np.ndarray, boundary_probs: list[np.ndarray]) -> np.ndarray:
    if not boundary_probs:
        return base_probs
    boundary_mean = np.mean(boundary_probs, axis=0)
    combined = base_probs.copy()
    amb_idx = LABEL2ID["Ambivalent"]
    nr_idx = LABEL2ID["Clear Non-Reply"]
    nonclear_mass = combined[:, amb_idx] + combined[:, nr_idx]
    combined[:, amb_idx] = nonclear_mass * boundary_mean[:, BOUNDARY_LABELS.index("Ambivalent")]
    combined[:, nr_idx] = nonclear_mass * boundary_mean[:, BOUNDARY_LABELS.index("Clear Non-Reply")]
    return combined / combined.sum(axis=1, keepdims=True)


def apply_clear_boundary(base_probs: np.ndarray, clear_probs: list[np.ndarray], alpha: float) -> np.ndarray:
    if not clear_probs:
        return base_probs
    clear_mean = np.mean(clear_probs, axis=0)
    clear_idx = LABEL2ID["Clear Reply"]
    amb_idx = LABEL2ID["Ambivalent"]
    nr_idx = LABEL2ID["Clear Non-Reply"]
    specialized = base_probs.copy()
    clear_mass = clear_mean[:, CLEAR_BOUNDARY_LABELS.index("Clear Reply")]
    nonclear_mass = clear_mean[:, CLEAR_BOUNDARY_LABELS.index("Non-Clear")]
    base_nonclear = specialized[:, amb_idx] + specialized[:, nr_idx]
    amb_ratio = np.divide(
        specialized[:, amb_idx],
        base_nonclear,
        out=np.full(len(specialized), 0.85, dtype=np.float64),
        where=base_nonclear > 1e-12,
    )
    specialized[:, clear_idx] = clear_mass
    specialized[:, amb_idx] = nonclear_mass * amb_ratio
    specialized[:, nr_idx] = nonclear_mass * (1.0 - amb_ratio)
    specialized = specialized / specialized.sum(axis=1, keepdims=True)
    alpha = min(1.0, max(0.0, alpha))
    blended = (1.0 - alpha) * base_probs + alpha * specialized
    return blended / blended.sum(axis=1, keepdims=True)


def apply_reply_boundary(base_probs: np.ndarray, reply_probs: list[np.ndarray], alpha: float) -> np.ndarray:
    if not reply_probs:
        return base_probs
    reply_mean = np.mean(reply_probs, axis=0)
    clear_idx = LABEL2ID["Clear Reply"]
    amb_idx = LABEL2ID["Ambivalent"]
    specialized = base_probs.copy()
    reply_mass = specialized[:, clear_idx] + specialized[:, amb_idx]
    specialized[:, clear_idx] = reply_mass * reply_mean[:, REPLY_BOUNDARY_LABELS.index("Clear Reply")]
    specialized[:, amb_idx] = reply_mass * reply_mean[:, REPLY_BOUNDARY_LABELS.index("Ambivalent")]
    specialized = specialized / specialized.sum(axis=1, keepdims=True)
    alpha = min(1.0, max(0.0, alpha))
    blended = (1.0 - alpha) * base_probs + alpha * specialized
    return blended / blended.sum(axis=1, keepdims=True)


def tune_bias(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    best_bias = np.zeros(probs.shape[1], dtype=np.float64)
    best_f1 = -1.0
    # A wider grid is still cheap on the internal dev set and avoids clipping the
    # non-reply calibration optimum for boundary-heavy ensembles.
    grid = np.round(np.arange(-3.0, 3.01, 0.05), 2)
    for amb_bias in grid:
        for nr_bias in grid:
            bias = np.array([0.0, amb_bias, nr_bias])
            pred = (logits + bias).argmax(axis=1)
            score = classification_metrics(y_true, pred, LABELS)["macro_f1"]
            if score > best_f1:
                best_f1 = score
                best_bias = bias
    return best_bias


def evaluate_probs(
    frame: pd.DataFrame,
    probs: np.ndarray,
    *,
    output_dir: Path,
    split_name: str,
    bias: np.ndarray | None = None,
) -> dict:
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    if bias is not None:
        logits = logits + bias
    preds = logits.argmax(axis=1)
    y_true = frame["label_id"].to_numpy()
    metrics = classification_metrics(y_true, preds, LABELS)
    write_json(metrics, output_dir / f"{split_name}_metrics.json")
    write_report(metrics, LABELS, output_dir / f"{split_name}_classification_report.csv")
    write_predictions(frame, probs, preds, LABELS, output_dir / f"{split_name}_predictions.csv")
    plot_confusion_matrix(metrics["confusion_matrix"], LABELS, output_dir / f"{split_name}_confusion.png")
    return metrics


def collect_probs(
    model_dirs: list[Path],
    boundary_dirs: list[Path],
    clear_dirs: list[Path],
    reply_dirs: list[Path],
    frame: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    base_probs = []
    for model_dir in model_dirs:
        probs, metadata = predict_model(
            model_dir / "best_model" if (model_dir / "best_model").exists() else model_dir,
            frame,
            batch_size=args.batch_size,
            device=device,
            fp16=args.fp16,
            chunked_inference=args.chunked_inference,
            chunk_size=args.chunk_size,
            chunk_stride=args.chunk_stride,
            max_chunks=args.max_chunks,
            chunk_aggregation=args.chunk_aggregation,
        )
        if metadata["task"] != "multiclass":
            raise ValueError(f"{model_dir} is not a multiclass model")
        base_probs.append(probs)
    mean_base = np.mean(base_probs, axis=0)

    boundary_probs = []
    clear_probs = []
    reply_probs = []
    for model_dir in clear_dirs:
        probs, metadata = predict_model(
            model_dir / "best_model" if (model_dir / "best_model").exists() else model_dir,
            frame,
            batch_size=args.batch_size,
            device=device,
            fp16=args.fp16,
            chunked_inference=args.chunked_inference,
            chunk_size=args.chunk_size,
            chunk_stride=args.chunk_stride,
            max_chunks=args.max_chunks,
            chunk_aggregation=args.chunk_aggregation,
        )
        if metadata["task"] != "clear_boundary":
            raise ValueError(f"{model_dir} is not a clear-boundary model")
        clear_probs.append(probs)
    mean_base = apply_clear_boundary(mean_base, clear_probs, args.clear_alpha)

    for model_dir in reply_dirs:
        probs, metadata = predict_model(
            model_dir / "best_model" if (model_dir / "best_model").exists() else model_dir,
            frame,
            batch_size=args.batch_size,
            device=device,
            fp16=args.fp16,
            chunked_inference=args.chunked_inference,
            chunk_size=args.chunk_size,
            chunk_stride=args.chunk_stride,
            max_chunks=args.max_chunks,
            chunk_aggregation=args.chunk_aggregation,
        )
        if metadata["task"] != "reply_boundary":
            raise ValueError(f"{model_dir} is not a reply-boundary model")
        reply_probs.append(probs)
    mean_base = apply_reply_boundary(mean_base, reply_probs, args.reply_alpha)

    for model_dir in boundary_dirs:
        probs, metadata = predict_model(
            model_dir / "best_model" if (model_dir / "best_model").exists() else model_dir,
            frame,
            batch_size=args.batch_size,
            device=device,
            fp16=args.fp16,
            chunked_inference=args.chunked_inference,
            chunk_size=args.chunk_size,
            chunk_stride=args.chunk_stride,
            max_chunks=args.max_chunks,
            chunk_aggregation=args.chunk_aggregation,
        )
        if metadata["task"] != "boundary":
            raise ValueError(f"{model_dir} is not a boundary model")
        boundary_probs.append(probs)
    return apply_boundary(mean_base, boundary_probs)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = get_splits(seed=args.seed, max_dev_samples=args.max_dev_samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_json(
        {
            "model_dirs": [str(path) for path in args.model_dirs],
            "boundary_model_dirs": [str(path) for path in args.boundary_model_dirs],
            "clear_model_dirs": [str(path) for path in args.clear_model_dirs],
            "reply_model_dirs": [str(path) for path in args.reply_model_dirs],
            "clear_alpha": args.clear_alpha,
            "reply_alpha": args.reply_alpha,
            "chunked_inference": args.chunked_inference,
            "chunk_size": args.chunk_size,
            "chunk_stride": args.chunk_stride,
            "max_chunks": args.max_chunks,
            "chunk_aggregation": args.chunk_aggregation,
            "max_dev_samples": args.max_dev_samples,
            "eval_final": args.eval_final,
        },
        args.output_dir / "ensemble_config.json",
    )

    dev_probs = collect_probs(
        args.model_dirs,
        args.boundary_model_dirs,
        args.clear_model_dirs,
        args.reply_model_dirs,
        splits.dev,
        args,
        device,
    )
    if args.bias_json:
        with args.bias_json.open("r", encoding="utf-8") as f:
            bias = np.asarray(json.load(f)["bias"], dtype=np.float64)
    else:
        bias = tune_bias(dev_probs, splits.dev["label_id"].to_numpy())
    write_json({"bias": bias.tolist(), "labels": LABELS}, args.output_dir / "calibration.json")
    dev_metrics = evaluate_probs(splits.dev, dev_probs, output_dir=args.output_dir, split_name="dev", bias=bias)
    print(f"dev_macro_f1={dev_metrics['macro_f1']:.6f}")

    if args.eval_final:
        final_probs = collect_probs(
            args.model_dirs,
            args.boundary_model_dirs,
            args.clear_model_dirs,
            args.reply_model_dirs,
            splits.final,
            args,
            device,
        )
        final_metrics = evaluate_probs(splits.final, final_probs, output_dir=args.output_dir, split_name="final", bias=bias)
        print(f"final_macro_f1={final_metrics['macro_f1']:.6f}")


if __name__ == "__main__":
    main()
