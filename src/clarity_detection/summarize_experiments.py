from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize QEvasion experiment artifacts.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiment_scoreboard"))
    return parser.parse_args()


def read_json(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def read_metric(path: Path, key: str = "macro_f1") -> float | None:
    payload = read_json(path)
    if not payload:
        return None
    value = payload.get(key)
    return float(value) if value is not None else None


def kind_for(name: str) -> str:
    lowered = name.lower()
    if "smoke" in lowered:
        return "smoke"
    if "blend" in lowered:
        return "blend"
    if "zero_shot" in lowered:
        return "zero_shot"
    if "decision_layer" in lowered or "conditional_bias" in lowered:
        return "decision_layer"
    if "comparison" in lowered or "analysis" in lowered or "adversarial" in lowered or "robustness" in lowered:
        return "analysis"
    if "ensemble" in lowered:
        return "ensemble"
    if "deberta" in lowered or "boundary" in lowered or "longformer" in lowered:
        return "model"
    return "other"


def cleanliness_for(name: str) -> str:
    lowered = name.lower()
    if lowered == "ensemble_final_frozen":
        return "course_clean_final_once"
    if "diag" in lowered or "final" in lowered or "blend" in lowered or "conditional" in lowered:
        return "diagnostic"
    if "smoke" in lowered:
        return "smoke"
    return "internal_dev_only"


def collect_run(path: Path) -> dict:
    name = path.name
    row = {
        "name": name,
        "kind": kind_for(name),
        "cleanliness": cleanliness_for(name),
        "dev_macro_f1": read_metric(path / "dev_metrics.json"),
        "dev_fit_macro_f1": read_metric(path / "dev_fit_metrics.json"),
        "final_macro_f1": read_metric(path / "final_metrics.json"),
        "accuracy_dev": read_metric(path / "dev_metrics.json", "accuracy"),
        "accuracy_final": read_metric(path / "final_metrics.json", "accuracy"),
        "best_dev_macro_f1": None,
        "best_epoch": None,
        "task": None,
        "model_name": None,
        "group_mode": None,
        "input_format": None,
        "notes": "",
        "trust_for_selection": "unknown",
    }
    summary = read_json(path / "training_summary.json")
    if summary:
        row["best_dev_macro_f1"] = summary.get("best_dev_macro_f1")
        row["best_epoch"] = summary.get("best_epoch")
    metadata = read_json(path / "metadata.json") or read_json(path / "best_model" / "metadata.json")
    if metadata:
        row["task"] = metadata.get("task")
        row["model_name"] = metadata.get("model_name")
        row["group_mode"] = metadata.get("group_mode")
        row["input_format"] = metadata.get("input_format", "pair")
        if metadata.get("train_on_full"):
            row["trust_for_selection"] = "inflated_train_on_full"
        elif metadata.get("max_train_samples") or metadata.get("max_dev_samples"):
            row["trust_for_selection"] = "smoke_sampled"
        else:
            row["trust_for_selection"] = "clean_internal_dev"
    blend = read_json(path / "blend_summary.json")
    if blend:
        weights = blend.get("weights", {})
        selected = [name for name, weight in weights.items() if float(weight) > 1e-6]
        row["notes"] = "blend_weights=" + ",".join(selected)
        row["trust_for_selection"] = "diagnostic_blend"
    zero = read_json(path / "zero_shot_config.json")
    if zero:
        row["model_name"] = zero.get("model_name")
        row["input_format"] = "zero_shot_" + str(zero.get("hypothesis_set"))
        row["trust_for_selection"] = "diagnostic_zero_shot"
    if row["kind"] == "smoke":
        row["trust_for_selection"] = "smoke_sampled"
    elif row["trust_for_selection"] == "unknown":
        if row["cleanliness"] == "course_clean_final_once":
            row["trust_for_selection"] = "course_clean"
        elif row["cleanliness"] == "diagnostic":
            row["trust_for_selection"] = "diagnostic_post_final"
        elif row["kind"] == "ensemble":
            row["trust_for_selection"] = "clean_internal_dev"
    return row


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    text = frame.copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            text[col] = text[col].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
        else:
            text[col] = text[col].map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(text.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(text.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in text.to_numpy(dtype=str)]
    return "\n".join([header, divider, *rows])


def top_table(frame: pd.DataFrame, metric: str, n: int = 20) -> pd.DataFrame:
    cols = []
    for col in [
        "name",
        "kind",
        "cleanliness",
        metric,
        "dev_macro_f1",
        "final_macro_f1",
        "task",
        "model_name",
        "trust_for_selection",
        "notes",
    ]:
        if col not in cols:
            cols.append(col)
    available_cols = [col for col in cols if col in frame.columns]
    return frame.dropna(subset=[metric]).sort_values(metric, ascending=False)[available_cols].head(n)


def write_report(frame: pd.DataFrame, output_dir: Path) -> None:
    best_clean = frame[frame["cleanliness"].eq("course_clean_final_once")]
    best_diag_final = frame[(frame["cleanliness"].eq("diagnostic")) & frame["final_macro_f1"].notna()]
    trustworthy_dev = frame[
        frame["trust_for_selection"].isin(["clean_internal_dev", "unknown"])
        & ~frame["kind"].isin(["smoke", "analysis"])
    ]
    lines = [
        "# QEvasion Experiment Scoreboard",
        "",
        "This report is generated from current files under `outputs/`.",
        "",
        "## Best Course-Clean Final Runs",
        "",
        markdown_table(top_table(best_clean, "final_macro_f1", 10)),
        "",
        "## Best Diagnostic Final Runs",
        "",
        markdown_table(top_table(best_diag_final, "final_macro_f1", 20)),
        "",
        "## Best Internal Dev Runs",
        "",
        markdown_table(top_table(frame, "dev_macro_f1", 20)),
        "",
        "## Best Trustworthy Internal Dev Runs",
        "",
        markdown_table(top_table(trustworthy_dev, "dev_macro_f1", 20)),
        "",
        "## Best Training Summaries",
        "",
        markdown_table(top_table(frame, "best_dev_macro_f1", 20)),
        "",
        "## Decision Notes",
        "",
        "- Treat `course_clean_final_once` as the only official-style result.",
        "- Treat `diagnostic` final metrics as post-final analysis, not course-clean model selection.",
        "- Current evidence says CSV-level blending and simple decision layers overfit internal dev.",
        "- The robust `reply_boundary` specialist strongly improves internal dev but does not transfer to final diagnostics.",
        "- Next priority: representation/architecture changes (`directness_nli`, `rubric_prompt`, then revised large LoRA).",
    ]
    (output_dir / "scoreboard.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(args.outputs_dir.iterdir()):
        if path.is_dir():
            rows.append(collect_run(path))
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "scoreboard.csv", index=False)
    write_report(frame, args.output_dir)
    best_final = frame.dropna(subset=["final_macro_f1"]).sort_values("final_macro_f1", ascending=False).head(1)
    if not best_final.empty:
        row = best_final.iloc[0]
        print(f"best_final={row['name']} macro_f1={row['final_macro_f1']:.6f} cleanliness={row['cleanliness']}")
    best_dev = frame.dropna(subset=["dev_macro_f1"]).sort_values("dev_macro_f1", ascending=False).head(1)
    if not best_dev.empty:
        row = best_dev.iloc[0]
        print(f"best_dev={row['name']} macro_f1={row['dev_macro_f1']:.6f} cleanliness={row['cleanliness']}")


if __name__ == "__main__":
    main()
