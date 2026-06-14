from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .data import GROUP_MODES, LABEL_COLUMN, get_splits, save_split_summary, set_seed
from .labels import (
    BOUNDARY_ID2LABEL,
    BOUNDARY_LABEL2ID,
    BOUNDARY_LABELS,
    CLEAR_BOUNDARY_ID2LABEL,
    CLEAR_BOUNDARY_LABEL2ID,
    CLEAR_BOUNDARY_LABELS,
    ID2LABEL,
    LABELS,
    REPLY_BOUNDARY_ID2LABEL,
    REPLY_BOUNDARY_LABEL2ID,
    REPLY_BOUNDARY_LABELS,
)
from .metrics import classification_metrics, write_json, write_report
from .plots import plot_confusion_matrix, plot_label_distribution


class QPairDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer,
        *,
        label2id: dict[str, int],
        max_length: int,
        truncation: str,
        input_format: str = "pair",
        max_question_tokens: int = 96,
        head_ratio: float = 0.7,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.truncation = truncation
        self.input_format = input_format
        self.max_question_tokens = max_question_tokens
        self.head_ratio = head_ratio

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict:
        row = self.frame.iloc[idx]
        question, answer = format_model_input(row["question"], row["interview_answer"], self.input_format)
        if self.truncation == "standard":
            enc = self.tokenizer(
                question,
                answer,
                truncation="only_second",
                max_length=self.max_length,
                padding=False,
            )
        elif self.truncation == "head_tail":
            enc = self._encode_head_tail(question, answer)
        else:
            raise ValueError(f"Unknown truncation: {self.truncation}")
        enc["labels"] = self.label2id.get(row[LABEL_COLUMN], 0)
        enc["sample_weight"] = float(row.get("sample_weight", 1.0))
        return enc

    def _encode_head_tail(self, question: str, answer: str) -> dict:
        q_ids = self._tokenize_no_special(question)
        a_ids = self._tokenize_no_special(answer)
        q_ids = q_ids[: self.max_question_tokens]

        # If even the question is too long, trim until pair special tokens fit.
        while q_ids and len(self._prepare_pair(q_ids, [])["input_ids"]) >= self.max_length:
            q_ids = q_ids[:-8]

        available_for_answer = self.max_length - len(self._prepare_pair(q_ids, [])["input_ids"])
        available_for_answer = max(0, available_for_answer)
        if len(a_ids) > available_for_answer:
            head = min(len(a_ids), int(math.ceil(available_for_answer * self.head_ratio)))
            tail = max(0, available_for_answer - head)
            a_ids = a_ids[:head] + (a_ids[-tail:] if tail > 0 else [])

        enc = self._prepare_pair(q_ids, a_ids)
        return {key: value for key, value in enc.items() if key in {"input_ids", "attention_mask", "token_type_ids"}}

    def _prepare_pair(self, q_ids: list[int], a_ids: list[int]) -> dict:
        if hasattr(self.tokenizer, "prepare_for_model"):
            return self.tokenizer.prepare_for_model(
                q_ids,
                pair_ids=a_ids,
                add_special_tokens=True,
                padding=False,
                truncation=False,
                return_attention_mask=True,
            )
        if hasattr(self.tokenizer, "build_inputs_with_special_tokens"):
            input_ids = self.tokenizer.build_inputs_with_special_tokens(q_ids, a_ids)
            enc = {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
            if "token_type_ids" in self.tokenizer.model_input_names and hasattr(
                self.tokenizer, "create_token_type_ids_from_sequences"
            ):
                enc["token_type_ids"] = self.tokenizer.create_token_type_ids_from_sequences(q_ids, a_ids)[
                    : len(input_ids)
                ]
            return enc

        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        tokenizer_name = type(self.tokenizer).__name__.lower()
        if cls_id is None or sep_id is None:
            raise ValueError("Tokenizer does not expose CLS/SEP ids needed for manual head_tail assembly.")

        if "roberta" in tokenizer_name or "longformer" in tokenizer_name:
            input_ids = [cls_id] + q_ids + [sep_id, sep_id] + a_ids + [sep_id]
            token_type_ids = [0] * len(input_ids)
        else:
            input_ids = [cls_id] + q_ids + [sep_id] + a_ids + [sep_id]
            token_type_ids = [0] * (len(q_ids) + 2) + [1] * (len(a_ids) + 1)

        enc = {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
        if "token_type_ids" in self.tokenizer.model_input_names:
            enc["token_type_ids"] = token_type_ids
        return enc

    def _tokenize_no_special(self, text: str) -> list[int]:
        try:
            return self.tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]
        except TypeError:
            return self.tokenizer(text, add_special_tokens=False)["input_ids"]


def format_model_input(question: str, answer: str, input_format: str) -> tuple[str, str]:
    if input_format == "pair":
        return question, answer
    if input_format == "task_prompt":
        return (
            "Question: "
            + question
            + "\nTask: Decide whether the answer is a Clear Reply, Ambivalent, or Clear Non-Reply.",
            answer,
        )
    if input_format == "rubric_prompt":
        return (
            "Question: "
            + question
            + "\nRubric: Clear Reply directly answers the question. Ambivalent partially answers, hedges, or is unclear. Clear Non-Reply refuses, redirects, or does not answer.",
            answer,
        )
    if input_format == "directness_nli":
        return (
            "Question: " + question + "\nClaim: The answer clearly and directly answers the question.",
            answer,
        )
    raise ValueError(f"Unknown input_format: {input_format}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train local transformer classifier.")
    parser.add_argument(
        "--task",
        choices=["multiclass", "boundary", "clear_boundary", "reply_boundary"],
        default="multiclass",
    )
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--split-seed", type=int, default=13)
    parser.add_argument("--group-mode", choices=GROUP_MODES, default="url_question_answer")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--sample-weight-mode", choices=["none", "qa_conflict_downweight", "answer_conflict_downweight"], default="none")
    parser.add_argument("--conflict-downweight", type=float, default=0.5)
    parser.add_argument("--sample-weight-csv", type=Path)
    parser.add_argument("--sample-weight-column", default="sample_weight")
    parser.add_argument("--missing-sample-weight", choices=["one", "error"], default="one")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--truncation", choices=["head_tail", "standard"], default="head_tail")
    parser.add_argument(
        "--input-format",
        choices=["pair", "task_prompt", "rubric_prompt", "directness_nli"],
        default="pair",
    )
    parser.add_argument("--max-question-tokens", type=int, default=96)
    parser.add_argument("--head-ratio", type=float, default=0.7)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", nargs="*")
    parser.add_argument("--class-weight-mode", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--class-weight-multiplier", type=float)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-dev-samples", type=int)
    parser.add_argument("--train-on-full", action="store_true")
    parser.add_argument("--save-final-model", action="store_true")
    parser.add_argument("--eval-final", action="store_true")
    return parser.parse_args()


def label_maps(task: str) -> tuple[list[str], dict[str, int], dict[int, str]]:
    if task == "boundary":
        return BOUNDARY_LABELS, BOUNDARY_LABEL2ID, BOUNDARY_ID2LABEL
    if task == "clear_boundary":
        return CLEAR_BOUNDARY_LABELS, CLEAR_BOUNDARY_LABEL2ID, CLEAR_BOUNDARY_ID2LABEL
    if task == "reply_boundary":
        return REPLY_BOUNDARY_LABELS, REPLY_BOUNDARY_LABEL2ID, REPLY_BOUNDARY_ID2LABEL
    return LABELS, {label: idx for idx, label in enumerate(LABELS)}, ID2LABEL


def build_model(args: argparse.Namespace, num_labels: int, labels: list[str]):
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}
    common_kwargs = {
        "num_labels": num_labels,
        "id2label": id2label,
        "label2id": label2id,
        "ignore_mismatched_sizes": True,
    }
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            dtype=torch.float32,
            **common_kwargs,
        )
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            torch_dtype=torch.float32,
            **common_kwargs,
        )
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    if args.lora:
        from peft import LoraConfig, TaskType, get_peft_model

        lower_name = args.model_name.lower()
        if args.lora_target_modules:
            target_modules = args.lora_target_modules
        else:
            target_modules = ["query_proj", "value_proj"]
            if "roberta" in lower_name or "longformer" in lower_name:
                target_modules = ["query", "value"]
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    return model


def make_class_weights(
    frame: pd.DataFrame,
    label2id: dict[str, int],
    *,
    mode: str,
    multiplier: float = 1.0,
) -> torch.Tensor:
    if mode == "none":
        weights = np.ones(len(label2id), dtype=np.float32)
    else:
        y = frame[LABEL_COLUMN].map(label2id).to_numpy()
        classes = np.arange(len(label2id))
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y).astype(np.float32)
    if len(weights) == 2:
        weights[1] *= multiplier
    return torch.tensor(weights, dtype=torch.float32)


def parameter_summary(model) -> dict[str, float | int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_fraction": trainable / total if total else 0.0,
    }


def stable_text_key(*parts: str) -> str:
    import hashlib

    raw = "\n".join(str(part or "").lower().strip() for part in parts)
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()


def add_sample_weights(frame: pd.DataFrame, mode: str, downweight: float) -> pd.DataFrame:
    frame = frame.copy()
    frame["sample_weight"] = 1.0
    if mode == "none":
        return frame
    if mode == "qa_conflict_downweight":
        keys = [stable_text_key(q, a) for q, a in zip(frame["question"], frame["interview_answer"])]
    elif mode == "answer_conflict_downweight":
        keys = [stable_text_key(a) for a in frame["interview_answer"]]
    else:
        raise ValueError(f"Unknown sample weight mode: {mode}")
    tmp = pd.DataFrame({"key": keys, "label": frame[LABEL_COLUMN].to_numpy()})
    conflict_keys = set(tmp.groupby("key")["label"].nunique().loc[lambda values: values > 1].index)
    frame.loc[tmp["key"].isin(conflict_keys).to_numpy(), "sample_weight"] = float(downweight)
    return frame


def apply_external_sample_weights(
    frame: pd.DataFrame,
    weight_csv: Path | None,
    *,
    weight_column: str,
    missing: str,
) -> pd.DataFrame:
    if weight_csv is None:
        return frame
    weights = pd.read_csv(weight_csv)
    key_cols = ["question", "interview_answer", LABEL_COLUMN]
    missing_cols = [col for col in [*key_cols, weight_column] if col not in weights.columns]
    if missing_cols:
        raise ValueError(f"Sample-weight CSV is missing columns: {missing_cols}")
    external_column = "__external_sample_weight"
    weights = weights[key_cols + [weight_column]].drop_duplicates(key_cols).rename(columns={weight_column: external_column})
    merged = frame.merge(weights, on=key_cols, how="left", validate="many_to_one")
    missing_mask = merged[external_column].isna()
    if missing_mask.any() and missing == "error":
        raise ValueError(f"Missing external sample weights for {int(missing_mask.sum())} training rows.")
    external = merged[external_column].fillna(1.0).astype(float).clip(lower=0.0)
    merged["sample_weight"] = merged["sample_weight"].astype(float) * external
    return merged.drop(columns=[external_column])


def collate_batch(tokenizer, batch: list[dict]) -> dict[str, torch.Tensor]:
    labels = torch.tensor([item.pop("labels") for item in batch], dtype=torch.long)
    sample_weights = torch.tensor([item.pop("sample_weight", 1.0) for item in batch], dtype=torch.float32)
    enc = tokenizer.pad(batch, padding=True, return_tensors="pt")
    enc["labels"] = labels
    enc["sample_weight"] = sample_weights
    return enc


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def predict(
    model,
    loader: DataLoader,
    *,
    device: torch.device,
    use_fp16: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict", leave=False):
            batch = move_to_device(batch, device)
            labels = batch.pop("labels")
            batch.pop("sample_weight", None)
            ctx = torch.amp.autocast("cuda", dtype=torch.float16) if use_fp16 and device.type == "cuda" else nullcontext()
            with ctx:
                outputs = model(**batch)
            logits_list.append(outputs.logits.detach().float().cpu().numpy())
            labels_list.append(labels.detach().cpu().numpy())
    logits = np.concatenate(logits_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    probs = softmax(logits)
    preds = probs.argmax(axis=1)
    return probs, preds, labels


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def write_predictions(
    frame: pd.DataFrame,
    probs: np.ndarray,
    preds: np.ndarray,
    labels: list[str],
    path: Path,
) -> None:
    pred_frame = frame[["question", "interview_answer", LABEL_COLUMN]].copy()
    pred_frame["prediction"] = [labels[i] for i in preds]
    for idx, label in enumerate(labels):
        pred_frame[f"prob_{label}"] = probs[:, idx]
    pred_frame.to_csv(path, index=False)


def evaluate_and_write(
    model,
    frame: pd.DataFrame,
    loader: DataLoader,
    *,
    labels: list[str],
    device: torch.device,
    use_fp16: bool,
    split_name: str,
    output_dir: Path,
) -> dict:
    probs, preds, y_true = predict(model, loader, device=device, use_fp16=use_fp16)
    metrics = classification_metrics(y_true, preds, labels)
    write_json(metrics, output_dir / f"{split_name}_metrics.json")
    write_report(metrics, labels, output_dir / f"{split_name}_classification_report.csv")
    write_predictions(frame, probs, preds, labels, output_dir / f"{split_name}_predictions.csv")
    plot_confusion_matrix(metrics["confusion_matrix"], labels, output_dir / f"{split_name}_confusion.png")
    return metrics


def train() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args_for_json = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    torch.backends.cuda.matmul.allow_tf32 = True

    labels, label2id, _ = label_maps(args.task)
    splits = get_splits(
        seed=args.split_seed,
        task=args.task,
        group_mode=args.group_mode,
        max_train_samples=args.max_train_samples,
        max_dev_samples=args.max_dev_samples,
        train_on_full=args.train_on_full,
    )
    set_seed(args.seed)
    save_split_summary(splits, args.output_dir)
    plot_label_distribution(splits.train, LABEL_COLUMN, args.output_dir / "train_label_distribution.png")
    plot_label_distribution(splits.dev, LABEL_COLUMN, args.output_dir / "dev_label_distribution.png")
    train_frame = add_sample_weights(splits.train, args.sample_weight_mode, args.conflict_downweight)
    train_frame = apply_external_sample_weights(
        train_frame,
        args.sample_weight_csv,
        weight_column=args.sample_weight_column,
        missing=args.missing_sample_weight,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = build_model(args, len(labels), labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_ds = QPairDataset(
        train_frame,
        tokenizer,
        label2id=label2id,
        max_length=args.max_length,
        truncation=args.truncation,
        input_format=args.input_format,
        max_question_tokens=args.max_question_tokens,
        head_ratio=args.head_ratio,
    )
    dev_ds = QPairDataset(
        splits.dev,
        tokenizer,
        label2id=label2id,
        max_length=args.max_length,
        truncation=args.truncation,
        input_format=args.input_format,
        max_question_tokens=args.max_question_tokens,
        head_ratio=args.head_ratio,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(tokenizer, batch),
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(tokenizer, batch),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_updates = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)
    first_param_dtype = next(model.parameters()).dtype
    use_grad_scaler = args.fp16 and device.type == "cuda" and first_param_dtype == torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    class_weight_multiplier = (
        args.class_weight_multiplier
        if args.class_weight_multiplier is not None
        else (1.35 if args.task == "boundary" else 1.0)
    )
    class_weights = make_class_weights(
        splits.train,
        label2id,
        mode=args.class_weight_mode,
        multiplier=class_weight_multiplier,
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        reduction="none",
        label_smoothing=args.label_smoothing,
    )
    param_summary = parameter_summary(model)

    metadata = {
        "task": args.task,
        "model_name": args.model_name,
        "labels": labels,
        "label2id": label2id,
        "hyperparameters": args_for_json,
        "train_rows": len(splits.train),
        "dev_rows": len(splits.dev),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "max_length": args.max_length,
        "truncation": args.truncation,
        "input_format": args.input_format,
        "max_question_tokens": args.max_question_tokens,
        "head_ratio": args.head_ratio,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "label_smoothing": args.label_smoothing,
        "sample_weight_mode": args.sample_weight_mode,
        "conflict_downweight": args.conflict_downweight,
        "sample_weight_csv": str(args.sample_weight_csv) if args.sample_weight_csv else None,
        "sample_weight_column": args.sample_weight_column,
        "missing_sample_weight": args.missing_sample_weight,
        "sample_weight_min": float(train_frame["sample_weight"].min()),
        "sample_weight_max": float(train_frame["sample_weight"].max()),
        "sample_weight_mean": float(train_frame["sample_weight"].mean()),
        "sample_weight_downweighted_rows": int((train_frame["sample_weight"] < 1.0).sum()),
        "sample_weight_upweighted_rows": int((train_frame["sample_weight"] > 1.0).sum()),
        "warmup_steps": warmup_steps,
        "total_updates_planned": total_updates,
        "class_weights": class_weights.detach().cpu().tolist(),
        "class_weight_mode": args.class_weight_mode,
        "class_weight_multiplier": class_weight_multiplier,
        "class_weight_labels": labels,
        **param_summary,
        "use_grad_scaler": use_grad_scaler,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "group_mode": args.group_mode,
        "lora": args.lora,
        "gradient_checkpointing": args.gradient_checkpointing,
        "train_on_full": args.train_on_full,
        "trained_at_unix": time.time(),
    }
    write_json(metadata, args.output_dir / "metadata.json")

    best_f1 = -1.0
    best_epoch = -1
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(progress, start=1):
            batch = move_to_device(batch, device)
            labels_tensor = batch.pop("labels")
            sample_weight = batch.pop("sample_weight")
            ctx = torch.amp.autocast("cuda", dtype=torch.float16) if args.fp16 and device.type == "cuda" else nullcontext()
            with ctx:
                outputs = model(**batch)
                per_row_loss = criterion(outputs.logits, labels_tensor)
                loss = ((per_row_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)) / args.grad_accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running_loss += loss.item() * args.grad_accum
            if step % args.grad_accum == 0 or step == len(train_loader):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(loss=f"{running_loss / step:.4f}")

        dev_metrics = evaluate_and_write(
            model,
            splits.dev,
            dev_loader,
            labels=labels,
            device=device,
            use_fp16=args.fp16,
            split_name="dev",
            output_dir=args.output_dir,
        )
        history.append({"epoch": epoch, "dev_macro_f1": dev_metrics["macro_f1"], "loss": running_loss / len(train_loader)})
        pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
        print(f"epoch={epoch} dev_macro_f1={dev_metrics['macro_f1']:.6f}")

        if dev_metrics["macro_f1"] > best_f1:
            best_f1 = dev_metrics["macro_f1"]
            best_epoch = epoch
            bad_epochs = 0
            model.save_pretrained(args.output_dir / "best_model")
            tokenizer.save_pretrained(args.output_dir / "best_model")
            write_json({**metadata, "best_epoch": best_epoch, "best_dev_macro_f1": best_f1}, args.output_dir / "best_model" / "metadata.json")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early_stopping epoch={epoch} best_epoch={best_epoch} best_dev_macro_f1={best_f1:.6f}")
                break

    write_json({"best_epoch": best_epoch, "best_dev_macro_f1": best_f1, "history": history}, args.output_dir / "training_summary.json")

    if args.save_final_model:
        model.save_pretrained(args.output_dir / "final_model")
        tokenizer.save_pretrained(args.output_dir / "final_model")
        write_json(
            {
                **metadata,
                "saved_after_epoch": history[-1]["epoch"] if history else None,
                "best_epoch": best_epoch,
                "best_dev_macro_f1": best_f1,
            },
            args.output_dir / "final_model" / "metadata.json",
        )

    if args.eval_final:
        final_ds = QPairDataset(
            splits.final,
            tokenizer,
            label2id=label2id,
            max_length=args.max_length,
            truncation=args.truncation,
            input_format=args.input_format,
            max_question_tokens=args.max_question_tokens,
            head_ratio=args.head_ratio,
        )
        final_loader = DataLoader(
            final_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_batch(tokenizer, batch),
        )
        final_metrics = evaluate_and_write(
            model,
            splits.final,
            final_loader,
            labels=labels,
            device=device,
            use_fp16=args.fp16,
            split_name="final",
            output_dir=args.output_dir,
        )
        print(f"final_macro_f1={final_metrics['macro_f1']:.6f}")


if __name__ == "__main__":
    train()
