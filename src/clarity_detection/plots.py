from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_confusion_matrix(matrix: list[list[int]], labels: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_label_distribution(df: pd.DataFrame, label_col: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = df[label_col].value_counts().sort_index()
    plt.figure(figsize=(8, 4))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="deep", legend=False)
    plt.ylabel("Count")
    plt.xlabel("Label")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

