# Metric, report, JSON, and plotting helpers for experiment outputs.

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize


# Compute label-only classification metrics used across experiments.
def compute_classification_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

# Return NaN instead of failing when a metric is undefined for a split.
def safe_metric(metric_fn, *args, **kwargs):
    try:
        return float(metric_fn(*args, **kwargs))
    except ValueError:
        return np.nan

# Compute probability-based multiclass metrics when class coverage allows.
def compute_probability_metrics(y_true, y_prob, label_names) -> dict[str, float]:
    class_ids = np.arange(len(label_names))
    y_true_bin = label_binarize(y_true, classes=class_ids)
    if len(label_names) == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.column_stack([1 - y_true_bin[:, 0], y_true_bin[:, 0]])
    return {
        "macro_auroc": safe_metric(roc_auc_score, y_true_bin, y_prob, average="macro", multi_class="ovr"),
        "macro_auprc": safe_metric(average_precision_score, y_true_bin, y_prob, average="macro"),
    }

# Combine prediction and probability metrics into one flat dictionary.
def compute_metrics(y_true, y_pred, y_prob, label_names) -> dict[str, float]:
    metrics = compute_classification_metrics(y_true, y_pred)
    metrics.update(compute_probability_metrics(y_true, y_prob, label_names))
    return metrics


# Return a sklearn classification report and confusion matrix bundle.
def evaluate_predictions(y_true, y_pred, label_names) -> dict:
    report = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": cm,
    }

# Convert numpy-heavy metric payloads into values accepted by json.dump.
def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value

# Write a JSON artifact, creating parent directories as needed.
def save_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(payload), file, indent=2)

# Build a per-class report with all configured labels represented.
def classification_report_dict(y_true, y_pred, label_names):
    return classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(label_names)),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )


# Render and save a labeled confusion matrix image.
def save_confusion_matrix_figure(y_true, y_pred, label_names, save_path: Path, title: str = "Confusion Matrix") -> None:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(label_names)))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(label_names)))
    ax.set_yticks(np.arange(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

# Plot training and validation loss/accuracy curves from a history table.
def plot_history(history_df, title: str | None = None, save_path: Path | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="validation")
    axes[0].set_title(f"{title} Loss" if title else "Loss")
    axes[0].legend()
    axes[1].plot(history_df["epoch"], history_df["train_acc"], label="train")
    axes[1].plot(history_df["epoch"], history_df["val_acc"], label="validation")
    axes[1].set_title(f"{title} Accuracy" if title else "Accuracy")
    axes[1].legend()
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig
