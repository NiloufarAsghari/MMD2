from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .labels import LABELS
from .metrics import classification_metrics, write_json, write_report
from .plots import plot_confusion_matrix


KEY_COLUMNS = ["question", "interview_answer", "clarity_label"]


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=PATH")
    name, path = value.split("=", 1)
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune a weighted blend of prediction CSV probabilities.")
    parser.add_argument("--dev", type=parse_named_path, action="append", required=True)
    parser.add_argument("--final", type=parse_named_path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blend-space", choices=["probability", "logit"], default="logit")
    parser.add_argument("--random-candidates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--bias-min", type=float, default=-3.0)
    parser.add_argument("--bias-max", type=float, default=3.0)
    parser.add_argument("--bias-step", type=float, default=0.05)
    parser.add_argument("--sample-weight-csv", type=Path)
    parser.add_argument("--sample-weight-column", default="final_like_score_oof")
    parser.add_argument("--sample-weight-floor", type=float, default=0.25)
    parser.add_argument("--sample-weight-power", type=float, default=1.0)
    return parser.parse_args()


def load_prediction_set(named_paths: list[tuple[str, Path]]) -> tuple[list[str], list[pd.DataFrame]]:
    names = []
    frames = []
    base_keys = None
    for name, path in named_paths:
        frame = pd.read_csv(path)
        keys = frame[KEY_COLUMNS].astype(str)
        if base_keys is None:
            base_keys = keys
        elif not keys.equals(base_keys):
            raise ValueError(f"{name} at {path} does not align row-for-row with the first prediction file.")
        names.append(name)
        frames.append(frame)
    return names, frames


def probs_from_frames(frames: list[pd.DataFrame], space: str) -> np.ndarray:
    prob_cols = [f"prob_{label}" for label in LABELS]
    probs = np.stack([frame[prob_cols].astype(float).to_numpy() for frame in frames], axis=0)
    probs = np.clip(probs, 1e-12, 1.0)
    if space == "logit":
        return np.log(probs)
    return probs


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values, 1e-12, None)
    return values / values.sum(axis=1, keepdims=True)


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def blend(stack: np.ndarray, weights: np.ndarray, space: str) -> np.ndarray:
    combined = np.tensordot(weights, stack, axes=(0, 0))
    if space == "logit":
        return softmax(combined)
    return normalize_rows(combined)


def label_ids(frame: pd.DataFrame) -> np.ndarray:
    label2id = {label: idx for idx, label in enumerate(LABELS)}
    return frame["clarity_label"].map(label2id).to_numpy()


def candidate_weights(count: int, n_models: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    weights = []
    weights.append(np.ones(n_models, dtype=np.float64) / n_models)
    for idx in range(n_models):
        one_hot = np.zeros(n_models, dtype=np.float64)
        one_hot[idx] = 1.0
        weights.append(one_hot)
    for concentration in [0.2, 0.5, 1.0, 2.0, 5.0]:
        draw_count = max(1, count // 5)
        weights.extend(rng.dirichlet(np.full(n_models, concentration), size=draw_count))
    return weights[: count + n_models + 1]


def load_sample_weights(reference: pd.DataFrame, args: argparse.Namespace) -> np.ndarray | None:
    if args.sample_weight_csv is None:
        return None
    weights_frame = pd.read_csv(args.sample_weight_csv)
    cols = KEY_COLUMNS + [args.sample_weight_column]
    missing = [col for col in cols if col not in weights_frame.columns]
    if missing:
        raise ValueError(f"Sample-weight CSV is missing columns: {missing}")
    weights_frame = weights_frame[cols].drop_duplicates(KEY_COLUMNS)
    merged = reference[KEY_COLUMNS].merge(weights_frame, on=KEY_COLUMNS, how="left", validate="many_to_one")
    if merged[args.sample_weight_column].isna().any():
        missing_count = int(merged[args.sample_weight_column].isna().sum())
        raise ValueError(f"Could not find sample weights for {missing_count} rows.")
    weights = merged[args.sample_weight_column].astype(float).to_numpy()
    weights = np.clip(weights, 0.0, None) ** args.sample_weight_power
    weights = args.sample_weight_floor + weights
    return weights / np.mean(weights)


def tune_weights(
    stack: np.ndarray,
    y_true: np.ndarray,
    *,
    space: str,
    random_candidates: int,
    seed: int,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    best_weights = np.ones(stack.shape[0], dtype=np.float64) / stack.shape[0]
    best_score = -1.0
    for weights in candidate_weights(random_candidates, stack.shape[0], seed):
        probs = blend(stack, weights, space)
        score = f1_score(y_true, probs.argmax(axis=1), average="macro", sample_weight=sample_weight)
        if score > best_score:
            best_score = float(score)
            best_weights = np.asarray(weights, dtype=np.float64)
    return best_weights, best_score


def tune_bias(
    probs: np.ndarray,
    y_true: np.ndarray,
    *,
    bias_min: float,
    bias_max: float,
    bias_step: float,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    grid = np.round(np.arange(bias_min, bias_max + (bias_step / 2), bias_step), 4)
    best_bias = np.zeros(len(LABELS), dtype=np.float64)
    best_score = -1.0
    for amb_bias in grid:
        for nr_bias in grid:
            bias = np.array([0.0, amb_bias, nr_bias])
            preds = (logits + bias).argmax(axis=1)
            score = f1_score(y_true, preds, average="macro", sample_weight=sample_weight)
            if score > best_score:
                best_score = float(score)
                best_bias = bias
    return best_bias, best_score


def apply_bias(probs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return softmax(np.log(np.clip(probs, 1e-12, 1.0)) + bias)


def write_predictions(frame: pd.DataFrame, probs: np.ndarray, output_dir: Path, split: str) -> dict:
    preds = probs.argmax(axis=1)
    y_true = label_ids(frame)
    metrics = classification_metrics(y_true, preds, LABELS)
    out = frame[KEY_COLUMNS].copy()
    out["prediction"] = [LABELS[idx] for idx in preds]
    for idx, label in enumerate(LABELS):
        out[f"prob_{label}"] = probs[:, idx]
    out.to_csv(output_dir / f"{split}_predictions.csv", index=False)
    write_json(metrics, output_dir / f"{split}_metrics.json")
    write_report(metrics, LABELS, output_dir / f"{split}_classification_report.csv")
    plot_confusion_matrix(metrics["confusion_matrix"], LABELS, output_dir / f"{split}_confusion.png")
    return metrics


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    names, dev_frames = load_prediction_set(args.dev)
    final_by_name = dict(args.final)
    final_frames = []
    if args.final:
        missing = [name for name in names if name not in final_by_name]
        if missing:
            raise ValueError(f"Missing final prediction paths for: {missing}")
        _, final_frames = load_prediction_set([(name, final_by_name[name]) for name in names])

    y_dev = label_ids(dev_frames[0])
    sample_weight = load_sample_weights(dev_frames[0], args)
    dev_stack = probs_from_frames(dev_frames, args.blend_space)
    weights, raw_dev_score = tune_weights(
        dev_stack,
        y_dev,
        space=args.blend_space,
        random_candidates=args.random_candidates,
        seed=args.seed,
        sample_weight=sample_weight,
    )
    dev_probs = blend(dev_stack, weights, args.blend_space)
    bias, biased_dev_score = tune_bias(
        dev_probs,
        y_dev,
        bias_min=args.bias_min,
        bias_max=args.bias_max,
        bias_step=args.bias_step,
        sample_weight=sample_weight,
    )
    dev_probs = apply_bias(dev_probs, bias)
    dev_metrics = write_predictions(dev_frames[0], dev_probs, args.output_dir, "dev")

    summary = {
        "names": names,
        "blend_space": args.blend_space,
        "weights": {name: float(weight) for name, weight in zip(names, weights)},
        "raw_dev_macro_f1_before_bias": raw_dev_score,
        "dev_macro_f1_after_bias_search": biased_dev_score,
        "bias": bias.tolist(),
        "sample_weight_csv": str(args.sample_weight_csv) if args.sample_weight_csv else None,
        "sample_weight_column": args.sample_weight_column if args.sample_weight_csv else None,
        "sample_weight_floor": args.sample_weight_floor if args.sample_weight_csv else None,
        "sample_weight_power": args.sample_weight_power if args.sample_weight_csv else None,
        "dev": dev_metrics,
    }

    if final_frames:
        final_stack = probs_from_frames(final_frames, args.blend_space)
        final_probs = apply_bias(blend(final_stack, weights, args.blend_space), bias)
        summary["final"] = write_predictions(final_frames[0], final_probs, args.output_dir, "final")

    write_json(summary, args.output_dir / "blend_summary.json")
    pd.DataFrame(
        [{"name": name, "weight": float(weight)} for name, weight in zip(names, weights)]
    ).to_csv(args.output_dir / "blend_weights.csv", index=False)
    print(f"dev_macro_f1={dev_metrics['macro_f1']:.6f}")
    if "final" in summary:
        print(f"final_macro_f1={summary['final']['macro_f1']:.6f}")


if __name__ == "__main__":
    main()
