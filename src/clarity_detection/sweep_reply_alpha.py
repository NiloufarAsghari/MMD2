from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import get_splits
from .ensemble import (
    apply_boundary,
    apply_clear_boundary,
    apply_reply_boundary,
    evaluate_probs,
    predict_model,
    tune_bias,
)
from .labels import LABELS
from .metrics import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep reply-boundary alpha with cached model probabilities.")
    parser.add_argument("--model-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--boundary-model-dirs", nargs="*", type=Path, default=[])
    parser.add_argument("--clear-model-dirs", nargs="*", type=Path, default=[])
    parser.add_argument("--reply-model-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--clear-alpha", type=float, default=0.35)
    parser.add_argument("--alphas", nargs="+", type=float, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bias-json", type=Path)
    parser.add_argument("--tune-bias", action="store_true")
    parser.add_argument("--eval-final", action="store_true")
    parser.add_argument("--max-dev-samples", type=int)
    parser.add_argument("--chunked-inference", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=384)
    parser.add_argument("--chunk-stride", type=int, default=192)
    parser.add_argument("--max-chunks", type=int, default=6)
    parser.add_argument("--chunk-aggregation", choices=["mean", "max", "mean_max", "noisy_or"], default="mean_max")
    return parser.parse_args()


def resolve_model_dir(model_dir: Path) -> Path:
    best_model = model_dir / "best_model"
    return best_model if best_model.exists() else model_dir


def predict_checked(
    model_dir: Path,
    frame: pd.DataFrame,
    *,
    expected_task: str,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    probs, metadata = predict_model(
        resolve_model_dir(model_dir),
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
    if metadata["task"] != expected_task:
        raise ValueError(f"{model_dir} task={metadata['task']!r}, expected {expected_task!r}")
    return probs


def collect_components(frame: pd.DataFrame, args: argparse.Namespace, device: torch.device) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    base_probs = [
        predict_checked(model_dir, frame, expected_task="multiclass", args=args, device=device)
        for model_dir in args.model_dirs
    ]
    mean_base = np.mean(base_probs, axis=0)

    clear_probs = [
        predict_checked(model_dir, frame, expected_task="clear_boundary", args=args, device=device)
        for model_dir in args.clear_model_dirs
    ]
    clear_adjusted = apply_clear_boundary(mean_base, clear_probs, args.clear_alpha)

    reply_probs = [
        predict_checked(model_dir, frame, expected_task="reply_boundary", args=args, device=device)
        for model_dir in args.reply_model_dirs
    ]
    boundary_probs = [
        predict_checked(model_dir, frame, expected_task="boundary", args=args, device=device)
        for model_dir in args.boundary_model_dirs
    ]
    return clear_adjusted, reply_probs, boundary_probs


def read_bias(args: argparse.Namespace) -> np.ndarray | None:
    if args.bias_json is None:
        return None
    with args.bias_json.open("r", encoding="utf-8") as f:
        return np.asarray(json.load(f)["bias"], dtype=np.float64)


def alpha_label(alpha: float) -> str:
    text = f"{alpha:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def evaluate_alpha(
    frame: pd.DataFrame,
    clear_adjusted: np.ndarray,
    reply_probs: list[np.ndarray],
    boundary_probs: list[np.ndarray],
    *,
    alpha: float,
    bias: np.ndarray,
    output_dir: Path,
    split_name: str,
) -> dict:
    probs = apply_reply_boundary(clear_adjusted, reply_probs, alpha)
    probs = apply_boundary(probs, boundary_probs)
    return evaluate_probs(frame, probs, output_dir=output_dir, split_name=split_name, bias=bias)


def main() -> None:
    args = parse_args()
    if args.bias_json and args.tune_bias:
        raise ValueError("--bias-json and --tune-bias are mutually exclusive")

    splits = get_splits(seed=args.seed, max_dev_samples=args.max_dev_samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_clear_adjusted, dev_reply_probs, dev_boundary_probs = collect_components(splits.dev, args, device)
    fixed_bias = read_bias(args)

    final_components = None
    if args.eval_final:
        final_components = collect_components(splits.final, args, device)

    rows = []
    for alpha in args.alphas:
        output_dir = Path(f"{args.output_prefix}{alpha_label(alpha)}")
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            {
                "model_dirs": [str(path) for path in args.model_dirs],
                "boundary_model_dirs": [str(path) for path in args.boundary_model_dirs],
                "clear_model_dirs": [str(path) for path in args.clear_model_dirs],
                "reply_model_dirs": [str(path) for path in args.reply_model_dirs],
                "clear_alpha": args.clear_alpha,
                "reply_alpha": alpha,
                "bias_mode": "tuned_internal_dev" if args.tune_bias else "fixed_json",
                "bias_json": str(args.bias_json) if args.bias_json else None,
                "eval_final": args.eval_final,
            },
            output_dir / "ensemble_config.json",
        )

        dev_probs = apply_boundary(
            apply_reply_boundary(dev_clear_adjusted, dev_reply_probs, alpha),
            dev_boundary_probs,
        )
        bias = tune_bias(dev_probs, splits.dev["label_id"].to_numpy()) if args.tune_bias else fixed_bias
        if bias is None:
            bias = np.zeros(len(LABELS), dtype=np.float64)
        write_json({"bias": bias.tolist(), "labels": LABELS}, output_dir / "calibration.json")

        dev_metrics = evaluate_probs(splits.dev, dev_probs, output_dir=output_dir, split_name="dev", bias=bias)
        row = {
            "alpha": alpha,
            "output_dir": str(output_dir),
            "dev_macro_f1": dev_metrics["macro_f1"],
            "dev_accuracy": dev_metrics["accuracy"],
            "bias": json.dumps(bias.tolist()),
        }
        if final_components is not None:
            final_metrics = evaluate_alpha(
                splits.final,
                *final_components,
                alpha=alpha,
                bias=bias,
                output_dir=output_dir,
                split_name="final",
            )
            row["final_macro_f1"] = final_metrics["macro_f1"]
            row["final_accuracy"] = final_metrics["accuracy"]
        rows.append(row)
        print(f"alpha={alpha:.3f} dev_macro_f1={dev_metrics['macro_f1']:.6f}")

    summary = pd.DataFrame(rows).sort_values("dev_macro_f1", ascending=False)
    summary_path = Path(f"{args.output_prefix}summary.csv")
    summary.to_csv(summary_path, index=False)
    best = summary.iloc[0]
    print(f"best_alpha={best['alpha']:.3f} dev_macro_f1={best['dev_macro_f1']:.6f} output_dir={best['output_dir']}")


if __name__ == "__main__":
    main()
