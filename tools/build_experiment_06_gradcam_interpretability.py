# Build the Experiment 06 Grad-CAM interpretability notebook.

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = PROJECT_ROOT / "Experiments" / "04_Xray_Text_Encoder_Benchmark.ipynb"
TARGET_NOTEBOOK = PROJECT_ROOT / "Experiments" / "06_Xray_GradCAM_Interpretability.ipynb"


# Convert a multiline string into notebook source-line format.
def to_source(text: str) -> list[str]:
    text = text.strip("\n") + "\n"
    return list(text.splitlines(keepends=True))


# Build a markdown cell dictionary for nbformat JSON.
def markdown_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": to_source(text)}


# Build an unexecuted code cell dictionary for nbformat JSON.
def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": to_source(text),
    }


CELL_0 = """
# Experiment 6: Grad-CAM Interpretability for CheXbert Fusion

## Goal

This final experiment does not train a new model. Instead, it explains the behavior of the best multimodal model from experiment 4 by visualizing where the frontal image branch is focusing.

The notebook:

- reloads the saved CheXbert fusion checkpoint from experiment 4
- uses the same dataset splits, preprocessing, image encoder, and fusion architecture
- generates Grad-CAM heatmaps for the frontal image branch
- compares attention patterns for correct and incorrect predictions
- saves report-ready example overlays and summary CSVs

## Why This Experiment

Experiment 5 showed that lung-focused preprocessing changes which classes improve and which get worse. This notebook helps interpret those findings by showing whether the model is relying on localized frontal image regions or broader context.
"""


CELL_1 = """
## Imports and configuration
"""


CELL_2 = """
from pathlib import Path
import gc
import json
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from transformers import AutoTokenizer, BertConfig, BertModel
from transformers.utils.hub import cached_file

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

project_root = Path.cwd().resolve()
if not (project_root / "splits").exists():
    project_root = project_root.parent

base_dir = project_root
split_dir = base_dir / "splits"
train_csv = split_dir / "train.csv"
val_csv = split_dir / "val.csv"
test_csv = split_dir / "test.csv"

HF_CACHE_DIR = base_dir / "hf_cache"
IMG_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 0
MAX_LEN = 64
TOKENIZER_NAME = "emilyalsentzer/Bio_ClinicalBERT"

CHECKPOINT_PATH = (
    base_dir
    / "Experiments"
    / "outputs"
    / "experiment_02_text_encoder_benchmark"
    / "chexbert_fusion"
    / "best_checkpoint.pt"
)

EXPERIMENT_NAME = "experiment_06_gradcam_interpretability"
OUTPUT_DIR = base_dir / "Experiments" / "outputs" / EXPERIMENT_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HEATMAP_DIR = OUTPUT_DIR / "heatmaps"
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

print("Checkpoint:", CHECKPOINT_PATH)
print("Outputs:", OUTPUT_DIR)
"""


CELL_3 = """
## Load splits, labels, tokenizer, and dataset
"""


CELL_4 = """
train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)
test_df = pd.read_csv(test_csv)

label_encoder = LabelEncoder()
train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["label_enc"] = label_encoder.fit_transform(train_df["clean_label"])
val_df["label_enc"] = label_encoder.transform(val_df["clean_label"])
test_df["label_enc"] = label_encoder.transform(test_df["clean_label"])

label_names = list(label_encoder.classes_)
num_classes = len(label_names)

image_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_NAME,
    cache_dir=HF_CACHE_DIR,
)


class XRayMultimodalDataset(Dataset):
    def __init__(self, df, tokenizer, image_transform=None, max_len=MAX_LEN):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frontal = Image.open(row["frontal_path"]).convert("RGB")
        lateral = Image.open(row["lateral_path"]).convert("RGB")

        if self.image_transform is not None:
            frontal = self.image_transform(frontal)
            lateral = self.image_transform(lateral)

        text = str(row["indication"]) if pd.notna(row["indication"]) else ""
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "uid": row["uid"],
            "frontal_path": row["frontal_path"],
            "lateral_path": row["lateral_path"],
            "clean_label": row["clean_label"],
            "frontal": frontal,
            "lateral": lateral,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(row["label_enc"]), dtype=torch.long),
        }


test_loader = DataLoader(
    XRayMultimodalDataset(test_df, tokenizer, image_transform=image_transform),
    batch_size=1,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

print("Classes:", label_names)
print("Test samples:", len(test_df))
"""


CELL_5 = """
## Rebuild the CheXbert fusion model and load the saved checkpoint
"""


CELL_6 = """
def load_chexbert_backbone(repo_id, filename, cache_dir=None):
    checkpoint_path = cached_file(repo_id, filename, cache_dir=cache_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    bert_state = {
        key[len("bert."):]: value
        for key, value in state_dict.items()
        if key.startswith("bert.")
    }
    if not bert_state:
        raise ValueError("CheXbert checkpoint did not contain bert.* weights.")

    bert = BertModel(BertConfig())
    bert.load_state_dict(bert_state, strict=False)
    return bert


class DualImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.output_dim = feature_dim * 2

    def forward(self, frontal, lateral):
        frontal_features = self.backbone(frontal)
        lateral_features = self.backbone(lateral)
        return torch.cat([frontal_features, lateral_features], dim=1)


class CheXbertFusionClassifier(nn.Module):
    def __init__(self, num_classes, cache_dir=None):
        super().__init__()
        self.image_encoder = DualImageEncoder()
        self.bert = load_chexbert_backbone(
            "StanfordAIMI/RRG_scorers",
            "chexbert.pth",
            cache_dir=cache_dir,
        )
        hidden_size = self.bert.config.hidden_size
        self.text_proj = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.image_encoder.output_dim + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, frontal, lateral, input_ids, attention_mask):
        image_features = self.image_encoder(frontal, lateral)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        text_features = self.text_proj(cls_embedding)
        fused_features = torch.cat([image_features, text_features], dim=1)
        logits = self.classifier(fused_features)
        return logits


model = CheXbertFusionClassifier(num_classes=num_classes, cache_dir=HF_CACHE_DIR).to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

target_conv = model.image_encoder.backbone.layer4[-1].conv2
print("Model loaded.")
"""


CELL_7 = """
## Grad-CAM utilities
"""


CELL_8 = """
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, frontal, lateral, input_ids, attention_mask, target_class=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(frontal, lateral, input_ids, attention_mask)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        score = logits[:, target_class].sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam, logits.detach()

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def tensor_to_image(tensor):
    array = tensor.detach().cpu().permute(1, 2, 0).numpy()
    array = np.clip(array, 0.0, 1.0)
    return array


def overlay_heatmap(image_array, cam, alpha=0.45):
    cmap = plt.get_cmap("jet")
    heat = cmap(cam)[..., :3]
    overlay = (1 - alpha) * image_array + alpha * heat
    return np.clip(overlay, 0.0, 1.0)


def save_cam_panel(image_array, cam, overlay, save_path, title):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    axes[0].imshow(image_array)
    axes[0].set_title("Original Frontal")
    axes[0].axis("off")
    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")
    axes[2].imshow(overlay)
    axes[2].set_title(title)
    axes[2].axis("off")
    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)
"""


CELL_9 = """
## Run Grad-CAM on the test set and save outputs
"""


CELL_10 = """
grad_cam = GradCAM(model, target_conv)

records = []
all_true = []
all_pred = []

for batch in tqdm(test_loader, desc="Grad-CAM Test Sweep", dynamic_ncols=True):
    frontal = batch["frontal"].to(device)
    lateral = batch["lateral"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    cam, logits = grad_cam.generate(
        frontal=frontal,
        lateral=lateral,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    pred_idx = int(logits.argmax(dim=1).item())
    true_idx = int(labels.item())
    confidence = float(torch.softmax(logits, dim=1)[0, pred_idx].item())
    correct = pred_idx == true_idx

    frontal_image = tensor_to_image(batch["frontal"][0])
    overlay = overlay_heatmap(frontal_image, cam)

    label_slug = batch["clean_label"][0].replace(" ", "_")
    uid = str(batch["uid"][0])
    sample_dir = HEATMAP_DIR / ("correct" if correct else "incorrect") / label_slug
    sample_dir.mkdir(parents=True, exist_ok=True)
    save_path = sample_dir / f"{uid}_gradcam.png"

    save_cam_panel(
        frontal_image,
        cam,
        overlay,
        save_path,
        f"Pred: {label_names[pred_idx]} | True: {label_names[true_idx]}",
    )

    records.append(
        {
            "uid": uid,
            "true_label": label_names[true_idx],
            "pred_label": label_names[pred_idx],
            "correct": correct,
            "confidence": confidence,
            "cam_mean": float(cam.mean()),
            "cam_std": float(cam.std()),
            "cam_max": float(cam.max()),
            "cam_peak_row": int(np.unravel_index(np.argmax(cam), cam.shape)[0]),
            "cam_peak_col": int(np.unravel_index(np.argmax(cam), cam.shape)[1]),
            "frontal_path": batch["frontal_path"][0],
            "lateral_path": batch["lateral_path"][0],
            "heatmap_path": str(save_path),
        }
    )

    all_true.append(true_idx)
    all_pred.append(pred_idx)

grad_cam.close()

gradcam_df = pd.DataFrame(records)
gradcam_df.to_csv(OUTPUT_DIR / "gradcam_summary.csv", index=False)

cm = confusion_matrix(all_true, all_pred)
report = classification_report(
    all_true,
    all_pred,
    target_names=label_names,
    output_dict=True,
    zero_division=0,
)

pd.DataFrame(cm, index=label_names, columns=label_names).to_csv(
    OUTPUT_DIR / "test_confusion_matrix.csv"
)
with open(OUTPUT_DIR / "test_classification_report.json", "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)

print("Saved:", OUTPUT_DIR / "gradcam_summary.csv")
display(gradcam_df.head())
"""


CELL_11 = """
## Summarize correct vs incorrect attention patterns
"""


CELL_12 = """
summary_by_correctness = (
    gradcam_df.groupby("correct")[["cam_mean", "cam_std", "cam_max", "confidence"]]
    .mean()
    .reset_index()
)
summary_by_class = (
    gradcam_df.groupby(["true_label", "correct"])[["cam_mean", "cam_std", "cam_max", "confidence"]]
    .mean()
    .reset_index()
)

summary_by_correctness.to_csv(OUTPUT_DIR / "attention_summary_by_correctness.csv", index=False)
summary_by_class.to_csv(OUTPUT_DIR / "attention_summary_by_class.csv", index=False)

display(summary_by_correctness)
display(summary_by_class)
"""


CELL_13 = """
## Show representative examples for the report
"""


CELL_14 = """
correct_examples = (
    gradcam_df[gradcam_df["correct"]]
    .sort_values("confidence", ascending=False)
    .groupby("true_label")
    .head(1)
)
incorrect_examples = (
    gradcam_df[~gradcam_df["correct"]]
    .sort_values("confidence", ascending=False)
    .groupby("true_label")
    .head(1)
)

display(correct_examples[["uid", "true_label", "pred_label", "confidence", "heatmap_path"]])
display(incorrect_examples[["uid", "true_label", "pred_label", "confidence", "heatmap_path"]])
"""


CELL_15 = """
## Notes

- This notebook explains the best CheXbert fusion checkpoint from experiment 4 rather than training a new model.
- The generated heatmaps are frontal-branch Grad-CAM overlays, so they highlight image regions that most strongly influenced the predicted class score.
- These outputs are intended for report figures and qualitative analysis alongside the segmentation findings from experiment 5.
"""


# Assemble the ordered notebook cells for Experiment 06.
def build_cells() -> list[dict]:
    return [
        markdown_cell(CELL_0),
        markdown_cell(CELL_1),
        code_cell(CELL_2),
        markdown_cell(CELL_3),
        code_cell(CELL_4),
        markdown_cell(CELL_5),
        code_cell(CELL_6),
        markdown_cell(CELL_7),
        code_cell(CELL_8),
        markdown_cell(CELL_9),
        code_cell(CELL_10),
        markdown_cell(CELL_11),
        code_cell(CELL_12),
        markdown_cell(CELL_13),
        code_cell(CELL_14),
        markdown_cell(CELL_15),
    ]


# Write the generated Grad-CAM notebook to the Experiments folder.
def main() -> None:
    source_nb = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    target_nb = {
        "cells": build_cells(),
        "metadata": source_nb.get("metadata", {}),
        "nbformat": source_nb.get("nbformat", 4),
        "nbformat_minor": source_nb.get("nbformat_minor", 5),
    }
    TARGET_NOTEBOOK.write_text(
        json.dumps(target_nb, indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {TARGET_NOTEBOOK}")


if __name__ == "__main__":
    main()
