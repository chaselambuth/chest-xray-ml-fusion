# Patch or build the Experiment 03 gated-fusion ablation notebook.

from __future__ import annotations

import json
import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_NOTEBOOK = PROJECT_ROOT / "Experiments" / "03_Xray_ClinicalBERT_Gated_Fusion_Test.ipynb"
SOURCE_NOTEBOOK_CANDIDATES = [
    PROJECT_ROOT / "Experiments" / "03_Xray_ClinicalBERT_Gated_Fusion_Test.ipynb",
    PROJECT_ROOT / "Experiments" / "02_Xray_Text_Encoder_Selection_Reformatted.ipynb",
    PROJECT_ROOT / "Experiments" / "02_Xray_Text_Encoder_Selection.ipynb",
    PROJECT_ROOT / "Experiments" / "02_Xray_ClinicalBERT_Fusion.ipynb",
    PROJECT_ROOT / "Experiments" / "02_Outdated_Notebook.ipynb",
    PROJECT_ROOT / "Experiments" / "archive" / "02_Outdated_Notebook.ipynb",
]


# Pick the best available source notebook to update in place.
def resolve_source_notebook() -> Path:
    for candidate in SOURCE_NOTEBOOK_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find an Experiment 02 source notebook. Checked: "
        + ", ".join(str(path) for path in SOURCE_NOTEBOOK_CANDIDATES)
    )


# Convert a multiline string into notebook source-line format.
def to_source(code: str) -> list[str]:
    code = code.strip("\n") + "\n"
    return [line for line in code.splitlines(keepends=True)]


# Create a stable short id for an inserted notebook cell.
def cell_id(index: int, cell_type: str, source: str) -> str:
    digest = hashlib.sha1(f"{index}:{cell_type}:{source}".encode("utf-8")).hexdigest()
    return digest[:12]


CELL_0_MARKDOWN = """
# Experiment 3: Bio_ClinicalBERT Residual Gated Fusion Ablation

## Overview

This experiment keeps the current dataset splits, preprocessing, dual-view image encoder, and Bio_ClinicalBERT text encoder aligned with notebook 02.

The final comparison table reports four seed 42 results:

1. Bio_ClinicalBERT text-only baseline from notebook 02
2. Dual-view image-only baseline from notebook 02
3. Bio_ClinicalBERT standard fusion baseline from notebook 02
4. Bio_ClinicalBERT residual gated fusion trained in this notebook

## Fusion Focus

The multimodal branch replaces concatenation-only fusion with a residual gated block that keeps gated image features, gated text features, and their interaction term before classification.

The fusion code is organized around a pluggable fusion module so it can later be swapped for alternatives such as cross-attention.
"""


CELL_2_CODE = """
from pathlib import Path
import gc
import json
import math
import random
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from transformers import AutoTokenizer, AutoModel, BertConfig, BertModel
from transformers.utils.hub import cached_file

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Paths
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
EPOCHS = 3
LR_TEXT = 2e-5
LR_IMAGE = 1e-4
WEIGHT_DECAY = 1e-4
MAX_LEN = 64
FREEZE_BERT = True
FREEZE_IMAGE_BACKBONE = False
LR_SCHEDULE_NAME = "constant"

NOTEBOOK02_EXPERIMENT_NAME = "experiment_02_text_encoder_benchmark"
NOTEBOOK02_OUTPUT_DIR = base_dir / "Experiments" / "outputs" / NOTEBOOK02_EXPERIMENT_NAME
NOTEBOOK02_RUN_SUMMARY_CSV = NOTEBOOK02_OUTPUT_DIR / "run_summary.csv"
NOTEBOOK02_AGGREGATE_SUMMARY_CSV = NOTEBOOK02_OUTPUT_DIR / "aggregate_summary.csv"

EXPERIMENT_NAME = "experiment_03_gated_fusion_ablation"
OUTPUT_DIR = base_dir / "Experiments" / "outputs" / EXPERIMENT_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESOLVED_MODEL_CACHE_DIR = OUTPUT_DIR / "resolved_hf_models"
RESOLVED_MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
print("Saving artifacts to:", OUTPUT_DIR)

DEFAULT_TEXT_ENCODER_SPECS = [
    {
        "name": "Bio_ClinicalBERT",
        "slug": "bio_clinicalbert",
        "loader": "auto",
        "model_name": "emilyalsentzer/Bio_ClinicalBERT",
        "tokenizer_name": "emilyalsentzer/Bio_ClinicalBERT",
        "tokenizer_note": "Uses the tokenizer shipped with the Bio_ClinicalBERT checkpoint.",
        "description": "Clinical notes model initialized from BioBERT and trained on MIMIC notes.",
    },
]


JSON_CACHE_FILENAMES = {
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
}


def is_usable_cached_file(path):
    path = Path(path)
    if ".no_exist" in path.parts:
        return False
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if path.name in JSON_CACHE_FILENAMES:
        try:
            with open(path, "r", encoding="utf-8") as f:
                json.load(f)
        except json.JSONDecodeError:
            return False
    return True


def copy_if_needed(source_path, target_path):
    if not is_usable_cached_file(source_path):
        return False
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and target_path.stat().st_size == source_path.stat().st_size:
        return True
    shutil.copy2(source_path, target_path)
    return True


def hf_cache_repo_dir(repo_id, cache_dir=HF_CACHE_DIR):
    return Path(cache_dir) / f"models--{repo_id.replace('/', '--')}"


def first_cached_file(repo_id, filename_options, cache_dir=HF_CACHE_DIR):
    repo_dir = hf_cache_repo_dir(repo_id, cache_dir=cache_dir)
    if not repo_dir.exists():
        return None

    for filename in filename_options:
        candidates = list((repo_dir / "snapshots").glob(f"*/{filename}"))
        candidates.extend(repo_dir.glob(f"**/{filename}"))
        for candidate in candidates:
            if is_usable_cached_file(candidate):
                return candidate
    return None


def materialize_cached_repo(repo_id, required_files, optional_files=None, cache_dir=HF_CACHE_DIR):
    optional_files = optional_files or []
    target_dir = RESOLVED_MODEL_CACHE_DIR / repo_id.replace("/", "__")
    missing = []

    for filename_options in required_files:
        if isinstance(filename_options, str):
            filename_options = [filename_options]
        source_path = first_cached_file(repo_id, filename_options, cache_dir=cache_dir)
        if source_path is None:
            missing.append("/".join(filename_options))
            continue
        copy_if_needed(source_path, target_dir / source_path.name)

    for filename in optional_files:
        source_path = first_cached_file(repo_id, [filename], cache_dir=cache_dir)
        if source_path is not None:
            copy_if_needed(source_path, target_dir / filename)
        else:
            stale_path = target_dir / filename
            if stale_path.exists() and not is_usable_cached_file(stale_path):
                stale_path.unlink()

    if missing:
        print(f"Cached repo {repo_id} is missing: {missing}. Falling back to Hugging Face Hub.")
        return None

    print(f"Using materialized local cache for {repo_id}: {target_dir}")
    return target_dir


def resolve_model_source(model_name):
    return materialize_cached_repo(
        model_name,
        required_files=[
            "config.json",
            ["model.safetensors", "pytorch_model.bin"],
        ],
        optional_files=[
            "vocab.txt",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
    )


def resolve_tokenizer_source(tokenizer_name):
    return materialize_cached_repo(
        tokenizer_name,
        required_files=["vocab.txt"],
        optional_files=[
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
    )


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_checkpoint_payload(checkpoint_path):
    if checkpoint_path is None or pd.isna(checkpoint_path):
        return {}
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return {}
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def default_encoder_spec_from_slug(slug):
    for spec in DEFAULT_TEXT_ENCODER_SPECS:
        if spec["slug"] == slug:
            return dict(spec)
    raise KeyError(f"Unknown encoder slug from notebook 02: {slug}")


def load_notebook02_clinicalbert_context():
    if not NOTEBOOK02_RUN_SUMMARY_CSV.exists():
        raise FileNotFoundError(
            "Notebook 02 run summary was not found. Run notebook 02 first or provide "
            f"{NOTEBOOK02_RUN_SUMMARY_CSV}."
        )

    run_summary_df = pd.read_csv(NOTEBOOK02_RUN_SUMMARY_CSV)
    clinicalbert_slug = "bio_clinicalbert"
    fusion_runs = run_summary_df[
        run_summary_df["task_type"].eq("fusion")
        & run_summary_df["encoder_slug"].eq(clinicalbert_slug)
    ].copy()
    if fusion_runs.empty:
        raise ValueError("Notebook 02 run_summary.csv has no successful Bio_ClinicalBERT fusion runs.")

    for metric_col in ["val_accuracy", "val_macro_f1", "test_accuracy", "test_macro_f1"]:
        if metric_col in run_summary_df.columns:
            run_summary_df[metric_col] = pd.to_numeric(run_summary_df[metric_col], errors="coerce")
        if metric_col in fusion_runs.columns:
            fusion_runs[metric_col] = pd.to_numeric(fusion_runs[metric_col], errors="coerce")

    clinicalbert_group = {
        "encoder_name": "Bio_ClinicalBERT",
        "encoder_slug": clinicalbert_slug,
        "tokenizer_name": "emilyalsentzer/Bio_ClinicalBERT",
        "n_seeds": int(fusion_runs["seed"].nunique()),
        "seeds": ",".join(str(int(seed)) for seed in sorted(fusion_runs["seed"].dropna().unique())),
        "val_accuracy_mean": float(fusion_runs["val_accuracy"].mean()),
        "val_macro_f1_mean": float(fusion_runs["val_macro_f1"].mean()),
        "test_accuracy_mean": float(fusion_runs["test_accuracy"].mean()),
        "test_macro_f1_mean": float(fusion_runs["test_macro_f1"].mean()),
    }

    best_runs = fusion_runs.sort_values(
        ["val_accuracy", "val_macro_f1", "test_accuracy", "test_macro_f1"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    best_run = best_runs.iloc[0].to_dict()

    checkpoint = load_checkpoint_payload(best_run.get("checkpoint_path"))
    encoder_spec = checkpoint.get("encoder_spec") or default_encoder_spec_from_slug(clinicalbert_slug)
    if "tokenizer_name" not in encoder_spec or pd.isna(encoder_spec.get("tokenizer_name")):
        encoder_spec["tokenizer_name"] = clinicalbert_group["tokenizer_name"]

    inherited_settings = {
        "batch_size": int(checkpoint.get("batch_size", BATCH_SIZE)),
        "max_len": int(checkpoint.get("max_len", MAX_LEN)),
        "lr_schedule_name": checkpoint.get("lr_schedule_name", LR_SCHEDULE_NAME),
        "selection_metric": "fixed Bio_ClinicalBERT to match notebook 02 clinical baseline",
    }

    return {
        "encoder_spec": encoder_spec,
        "best_group": clinicalbert_group,
        "best_run": best_run,
        "selection_table": pd.DataFrame([clinicalbert_group]),
        "run_summary_df": run_summary_df,
        "inherited_settings": inherited_settings,
    }


NOTEBOOK02_SELECTION = load_notebook02_clinicalbert_context()
TEXT_ENCODER_SPEC = NOTEBOOK02_SELECTION["encoder_spec"]
NOTEBOOK02_BEST_GROUP = NOTEBOOK02_SELECTION["best_group"]
NOTEBOOK02_BEST_RUN = NOTEBOOK02_SELECTION["best_run"]
NOTEBOOK02_SELECTION_TABLE = NOTEBOOK02_SELECTION["selection_table"]
NOTEBOOK02_RUN_SUMMARY_DF = NOTEBOOK02_SELECTION["run_summary_df"]
INHERITED_NOTEBOOK02_SETTINGS = NOTEBOOK02_SELECTION["inherited_settings"]

BATCH_SIZE = INHERITED_NOTEBOOK02_SETTINGS["batch_size"]
MAX_LEN = INHERITED_NOTEBOOK02_SETTINGS["max_len"]
LR_SCHEDULE_NAME = INHERITED_NOTEBOOK02_SETTINGS["lr_schedule_name"]

SELECTED_ENCODER_NAME = TEXT_ENCODER_SPEC["name"]
SELECTED_ENCODER_SLUG = TEXT_ENCODER_SPEC["slug"]
TOKENIZER_NAME = TEXT_ENCODER_SPEC["tokenizer_name"]
TEXT_ONLY_RUN_SLUG = f"{SELECTED_ENCODER_SLUG}_text_only"
FUSION_RUN_SLUG = f"{SELECTED_ENCODER_SLUG}_residual_gated_fusion"

EXPERIMENT_SEEDS = [42]
SEED = 42
seed_everything(SEED)

print("Notebook 02 artifact directory:", NOTEBOOK02_OUTPUT_DIR)
print("Fixed notebook 03 encoder:", SELECTED_ENCODER_NAME)
print("Tokenizer:", TOKENIZER_NAME)
print("Notebook 03 seed:", SEED)
print("Gated fusion will run only for seed 42.")
print("Inherited settings:", INHERITED_NOTEBOOK02_SETTINGS)
"""


CELL_18_CODE = """
train_text_ds = ClinicalTextDataset(train_df, tokenizer)
val_text_ds = ClinicalTextDataset(val_df, tokenizer)
test_text_ds = ClinicalTextDataset(test_df, tokenizer)

train_img_ds = XRayDualImageDataset(train_df, image_transform=image_transform)
val_img_ds = XRayDualImageDataset(val_df, image_transform=image_transform)
test_img_ds = XRayDualImageDataset(test_df, image_transform=image_transform)

train_mm_ds = XRayMultimodalDataset(train_df, tokenizer, image_transform=image_transform)
val_mm_ds = XRayMultimodalDataset(val_df, tokenizer, image_transform=image_transform)
test_mm_ds = XRayMultimodalDataset(test_df, tokenizer, image_transform=image_transform)

train_text_loader = DataLoader(train_text_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_text_loader = DataLoader(val_text_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_text_loader = DataLoader(test_text_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

train_img_loader = DataLoader(train_img_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_img_loader = DataLoader(val_img_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_img_loader = DataLoader(test_img_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

train_mm_loader = DataLoader(train_mm_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_mm_loader = DataLoader(val_mm_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_mm_loader = DataLoader(test_mm_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

class_counts = train_df["clean_label"].value_counts()
class_weights = torch.tensor(
    [len(train_df) / class_counts[label] for label in label_names],
    dtype=torch.float32,
    device=device,
)

sample_batch = next(iter(train_mm_loader))
for key, value in sample_batch.items():
    print(key, tuple(value.shape))

print("Starting multimodal training setup...")
print("Selected text encoder:", SELECTED_ENCODER_NAME)
print("Using model cache:", HF_CACHE_DIR)


def get_cls_embedding(outputs):
    return outputs.last_hidden_state[:, 0, :]


def load_auto_backbone(model_name, cache_dir=None):
    local_source = resolve_model_source(model_name)
    model_source = local_source if local_source is not None else model_name
    local_only = local_source is not None
    print(f"Loading backbone from {model_source}...")
    try:
        return AutoModel.from_pretrained(
            model_source,
            cache_dir=cache_dir,
            local_files_only=local_only,
            use_safetensors=True,
        )
    except Exception as exc:
        print(f"Safetensors load failed for {model_name}: {exc}")
        return AutoModel.from_pretrained(
            model_source,
            cache_dir=cache_dir,
            local_files_only=local_only,
            use_safetensors=False,
        )


def load_chexbert_backbone(repo_id, filename, cache_dir=None):
    print(f"Loading CheXbert checkpoint from {repo_id}/{filename}...")
    checkpoint_path = cached_file(repo_id, filename, cache_dir=cache_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
    missing, unexpected = bert.load_state_dict(bert_state, strict=False)
    print("CheXbert backbone loaded.")
    if missing:
        print("Missing CheXbert keys:", missing[:10])
    if unexpected:
        print("Unexpected CheXbert keys:", unexpected[:10])
    return bert


def load_text_backbone(encoder_spec, cache_dir=None):
    if encoder_spec["loader"] == "auto":
        backbone = load_auto_backbone(encoder_spec["model_name"], cache_dir=cache_dir)
    elif encoder_spec["loader"] == "chexbert":
        backbone = load_chexbert_backbone(
            encoder_spec["repo_id"],
            encoder_spec["filename"],
            cache_dir=cache_dir,
        )
    else:
        raise ValueError(f"Unknown loader type: {encoder_spec['loader']}")

    hidden_size = backbone.config.hidden_size
    return backbone, hidden_size


class FusionBlockBase(nn.Module):
    def forward(self, image_features, text_features):
        raise NotImplementedError


class ResidualGatedFusionBlock(FusionBlockBase):
    def __init__(self, image_dim, text_dim, fusion_dim=256, dropout=0.2):
        super().__init__()
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid(),
        )
        self.output_dim = fusion_dim * 3

    def forward(self, image_features, text_features):
        image_hidden = self.image_proj(image_features)
        text_hidden = self.text_proj(text_features)
        gate = self.gate(torch.cat([image_hidden, text_hidden], dim=1))

        gated_image = gate * image_hidden
        gated_text = (1.0 - gate) * text_hidden
        interaction = gated_image * gated_text

        fused = torch.cat([gated_image, gated_text, interaction], dim=1)
        gate_weights = torch.stack(
            [gate.mean(dim=1), (1.0 - gate).mean(dim=1)],
            dim=1,
        )

        return {
            "fused_features": fused,
            "image_hidden": image_hidden,
            "text_hidden": text_hidden,
            "gate_weights": gate_weights,
        }


class TextEncoderClassifier(nn.Module):
    def __init__(self, encoder_spec, num_classes, freeze_bert=False, cache_dir=None):
        super().__init__()

        self.encoder_spec = encoder_spec
        self.bert, hidden_size = load_text_backbone(encoder_spec, cache_dir=cache_dir)

        if freeze_bert:
            print(f"Freezing {encoder_spec['name']} parameters for text model...")
            for param in self.bert.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes),
        )

    def encode_text(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return get_cls_embedding(outputs)

    def forward(self, input_ids, attention_mask):
        cls_embedding = self.encode_text(input_ids, attention_mask)
        logits = self.classifier(cls_embedding)
        return logits


class DualImageEncoder(nn.Module):
    def __init__(self, freeze_backbone=False):
        super().__init__()

        print("Loading ResNet18 image backbone...")
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.output_dim = feature_dim * 2

        if freeze_backbone:
            print("Freezing image backbone...")
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, frontal, lateral):
        frontal_features = self.backbone(frontal)
        lateral_features = self.backbone(lateral)
        return torch.cat([frontal_features, lateral_features], dim=1)


class DualImageModel(nn.Module):
    def __init__(self, num_classes, freeze_backbone=False):
        super().__init__()

        print("Building Dual Image Model...")
        self.image_encoder = DualImageEncoder(freeze_backbone=freeze_backbone)
        self.classifier = nn.Sequential(
            nn.Linear(self.image_encoder.output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        print("Dual Image Model ready")

    def forward(self, frontal, lateral):
        image_features = self.image_encoder(frontal, lateral)
        return self.classifier(image_features)


class SelectedEncoderGatedFusionModel(nn.Module):
    def __init__(
        self,
        encoder_spec,
        num_classes,
        fusion_block=None,
        fusion_dim=256,
        fusion_dropout=0.2,
        freeze_bert=False,
        freeze_image_backbone=False,
        cache_dir=None,
    ):
        super().__init__()

        self.encoder_spec = encoder_spec
        self.image_encoder = DualImageEncoder(freeze_backbone=freeze_image_backbone)
        self.bert, hidden_size = load_text_backbone(encoder_spec, cache_dir=cache_dir)

        if freeze_bert:
            print(f"Freezing {encoder_spec['name']} parameters for fusion model...")
            for param in self.bert.parameters():
                param.requires_grad = False

        self.fusion_block = fusion_block or ResidualGatedFusionBlock(
            image_dim=self.image_encoder.output_dim,
            text_dim=hidden_size,
            fusion_dim=fusion_dim,
            dropout=fusion_dropout,
        )
        fusion_dim = getattr(
            self.fusion_block,
            "output_dim",
            getattr(self.fusion_block, "image_proj")[0].out_features,
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def encode_text(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return get_cls_embedding(outputs)

    def forward(self, frontal, lateral, input_ids, attention_mask, return_aux=False):
        image_features = self.image_encoder(frontal, lateral)
        text_features = self.encode_text(input_ids, attention_mask)
        fusion_outputs = self.fusion_block(image_features, text_features)
        logits = self.classifier(fusion_outputs["fused_features"])

        if return_aux:
            return logits, fusion_outputs
        return logits


def configure_optimizer(model, lr_text=2e-5, lr_image=1e-4, weight_decay=1e-4):
    print("Configuring optimizer parameter groups...")

    bert_params = []
    image_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "bert" in name:
            bert_params.append(param)
        elif "image_encoder" in name or "image_backbone" in name or "backbone" in name:
            image_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if bert_params:
        param_groups.append({"params": bert_params, "lr": lr_text})
    if image_params:
        param_groups.append({"params": image_params, "lr": lr_image})
    if other_params:
        param_groups.append({"params": other_params, "lr": lr_image})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    print(
        f"Optimizer ready | "
        f"BERT groups: {len(bert_params)} | "
        f"Image groups: {len(image_params)} | "
        f"Other groups: {len(other_params)}"
    )
    return optimizer


def plot_history(history_df, title, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    axes[0].set_title(f"{title} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["train_acc"], label="train_acc")
    axes[1].plot(history_df["epoch"], history_df["val_acc"], label="val_acc")
    axes[1].set_title(f"{title} Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


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


def save_json(payload, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2)


def evaluate_predictions(y_true, y_pred, label_names):
    return {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def save_confusion_matrix_figure(cm, title, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)

    for i in range(len(label_names)):
        for j in range(len(label_names)):
            ax.text(j, i, cm[i][j], ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def save_model_artifacts(model_name, history_df, y_true, y_pred, output_dir, title):
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    history_df.to_csv(model_dir / "training_history.csv", index=False)
    plot_history(history_df, title, save_path=model_dir / "training_curves.png")

    metrics = evaluate_predictions(y_true, y_pred, label_names)
    cm = metrics["confusion_matrix"]

    save_json(
        {
            "model": model_name,
            "test_accuracy": metrics["test_accuracy"],
            "macro_f1": metrics["macro_f1"],
        },
        model_dir / "metrics.json",
    )
    save_json(metrics["classification_report"], model_dir / "classification_report.json")
    pd.DataFrame(cm, index=label_names, columns=label_names).to_csv(
        model_dir / "confusion_matrix.csv"
    )
    save_confusion_matrix_figure(
        cm,
        f"{title} Confusion Matrix",
        model_dir / "confusion_matrix.png",
    )

    report_df = pd.DataFrame(metrics["classification_report"]).transpose()
    display(report_df)

    print(f"{title} Test Accuracy:", round(metrics["test_accuracy"], 4))
    print(f"{title} Macro F1:", round(metrics["macro_f1"], 4))
    print()
    print("Classification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=label_names,
            zero_division=0,
        )
    )

    return metrics


def train_epoch_text(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    progress_bar = tqdm(loader, desc="Train Text", leave=False, dynamic_ncols=True)
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_count += batch_size

        progress_bar.set_postfix({
            "loss": f"{total_loss / total_count:.4f}",
            "acc": f"{total_correct / total_count:.4f}",
        })

    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def evaluate_text(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    y_true, y_pred = [], []

    progress_bar = tqdm(loader, desc="Eval Text", leave=False, dynamic_ncols=True)
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_count += batch_size

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

        progress_bar.set_postfix({
            "loss": f"{total_loss / total_count:.4f}",
            "acc": f"{total_correct / total_count:.4f}",
        })

    return total_loss / total_count, total_correct / total_count, np.array(y_true), np.array(y_pred)


print("Model, metric, and training utilities are ready.")
print("Notebook 03 will train only the residual gated fusion model for seed 42.")
"""


CELL_6_CODE = """
print("Notebook 02 Bio_ClinicalBERT context:")
display(NOTEBOOK02_SELECTION_TABLE)

print("Fixed encoder spec:")
print(json.dumps(TEXT_ENCODER_SPEC, indent=2))

print("Inherited notebook 02 settings used in this notebook:")
print(json.dumps(INHERITED_NOTEBOOK02_SETTINGS, indent=2))
"""


CELL_13_MARKDOWN = """
### Load the selected notebook 02 tokenizer
"""


CELL_14_CODE = """
def load_tokenizer_for_encoder(encoder_spec):
    tokenizer_name = encoder_spec["tokenizer_name"]
    local_source = resolve_tokenizer_source(tokenizer_name)
    tokenizer_source = local_source if local_source is not None else tokenizer_name
    local_only = local_source is not None
    print(f"Loading tokenizer for {encoder_spec['name']}: {tokenizer_source}")
    print("Tokenizer note:", encoder_spec.get("tokenizer_note", ""))
    return AutoTokenizer.from_pretrained(
        tokenizer_source,
        cache_dir=HF_CACHE_DIR,
        local_files_only=local_only,
    )


tokenizer = load_tokenizer_for_encoder(TEXT_ENCODER_SPEC)
print("Tokenizer loaded.")
"""


CELL_23_MARKDOWN = """
---

# **2. Notebook 02 Baselines**
"""


CELL_20_CODE = """
def aggregate_notebook02_runs(task_type, encoder_slug=None, model_label=None):
    df = NOTEBOOK02_RUN_SUMMARY_DF[NOTEBOOK02_RUN_SUMMARY_DF["task_type"].eq(task_type)].copy()
    if encoder_slug is not None:
        df = df[df["encoder_slug"].eq(encoder_slug)].copy()
    if df.empty:
        raise ValueError(f"No notebook 02 rows found for task_type={task_type}, encoder_slug={encoder_slug}")

    metric_cols = ["test_accuracy", "test_macro_f1", "val_accuracy", "val_macro_f1"]
    for metric_col in metric_cols:
        df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")

    return {
        "model": model_label or f"{encoder_slug}_{task_type}",
        "source": "notebook_02",
        "n_seeds": int(df["seed"].nunique()),
        "seeds": ",".join(str(int(seed)) for seed in sorted(df["seed"].dropna().unique())),
        "test_accuracy_mean": float(df["test_accuracy"].mean()),
        "test_accuracy_std": float(df["test_accuracy"].std(ddof=1)) if len(df) > 1 else 0.0,
        "test_macro_f1_mean": float(df["test_macro_f1"].mean()),
        "test_macro_f1_std": float(df["test_macro_f1"].std(ddof=1)) if len(df) > 1 else 0.0,
        "val_accuracy_mean": float(df["val_accuracy"].mean()),
        "val_macro_f1_mean": float(df["val_macro_f1"].mean()),
    }


notebook02_baseline_rows = [
    aggregate_notebook02_runs(
        "text_only",
        encoder_slug=SELECTED_ENCODER_SLUG,
        model_label=f"{SELECTED_ENCODER_NAME} text-only baseline",
    ),
    aggregate_notebook02_runs(
        "image_only",
        encoder_slug="dual_image_resnet18",
        model_label="Dual-view image baseline",
    ),
    aggregate_notebook02_runs(
        "fusion",
        encoder_slug=SELECTED_ENCODER_SLUG,
        model_label=f"{SELECTED_ENCODER_NAME} standard fusion baseline",
    ),
]

notebook02_baselines_df = pd.DataFrame(notebook02_baseline_rows)
notebook02_baselines_df.to_csv(OUTPUT_DIR / "notebook02_imported_baselines.csv", index=False)
display(notebook02_baselines_df)
"""


CELL_22_CODE = """
print("Imported notebook 02 baselines. Proceed to the gated fusion seed runs below.")
"""


CELL_24_CODE = """
print("Evaluation artifacts will be saved under:", OUTPUT_DIR)
"""


CELL_25_CODE = """
print("Text-only, image-only, and standard fusion baseline results are imported from notebook 02.")
"""


CELL_27_MARKDOWN = """
## Imported dual-view image baseline
"""


CELL_28_CODE = """
def train_epoch_image(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    progress = tqdm(loader, desc="Train Image", leave=False, dynamic_ncols=True)
    for batch in progress:
        frontal = batch["frontal"].to(device)
        lateral = batch["lateral"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(frontal, lateral)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

        progress.set_postfix(
            loss=f"{total_loss / total_count:.4f}",
            acc=f"{total_correct / total_count:.4f}",
        )

    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def evaluate_image(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    y_true, y_pred = [], []

    progress = tqdm(loader, desc="Eval Image", leave=False, dynamic_ncols=True)
    for batch in progress:
        frontal = batch["frontal"].to(device)
        lateral = batch["lateral"].to(device)
        labels = batch["labels"].to(device)

        logits = model(frontal, lateral)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

        progress.set_postfix(
            loss=f"{total_loss / total_count:.4f}",
            acc=f"{total_correct / total_count:.4f}",
        )

    return total_loss / total_count, total_correct / total_count, np.array(y_true), np.array(y_pred)


print("Image-only baseline is imported from notebook 02; image train/eval helpers remain available for reference.")
"""


CELL_29_CODE = """
display(notebook02_baselines_df[notebook02_baselines_df["model"].eq("Dual-view image baseline")])
"""


CELL_30_CODE = """
print("No image-only model retraining is performed in notebook 03.")
"""


CELL_30_CLEANUP_CODE = """
print("Imported baseline section complete.")
"""


CELL_31_MARKDOWN = """
## Residual gated multimodal fusion

This section keeps the same ResNet18 image encoder and selected notebook 02 text encoder, but swaps the notebook 02 concatenation baseline for a residual gated block that preserves image, text, and interaction features.
"""


CELL_32_CODE = """
def build_multimodal_loaders(seed):
    train_ds = XRayMultimodalDataset(train_df, tokenizer, image_transform=image_transform)
    val_ds = XRayMultimodalDataset(val_df, tokenizer, image_transform=image_transform)
    test_ds = XRayMultimodalDataset(test_df, tokenizer, image_transform=image_transform)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=generator,
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, val_loader, test_loader


def train_epoch_fusion(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    progress = tqdm(loader, desc="Train Fusion", leave=False, dynamic_ncols=True)
    for batch in progress:
        frontal = batch["frontal"].to(device)
        lateral = batch["lateral"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(frontal, lateral, input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

        progress.set_postfix(
            loss=f"{total_loss / total_count:.4f}",
            acc=f"{total_correct / total_count:.4f}",
        )

    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def evaluate_fusion(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    y_true, y_pred = [], []
    gate_weight_batches = []

    progress = tqdm(loader, desc="Eval Fusion", leave=False, dynamic_ncols=True)
    for batch in progress:
        frontal = batch["frontal"].to(device)
        lateral = batch["lateral"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits, aux = model(
            frontal,
            lateral,
            input_ids,
            attention_mask,
            return_aux=True,
        )
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        gate_weight_batches.append(aux["gate_weights"].cpu().numpy())

        progress.set_postfix(
            loss=f"{total_loss / total_count:.4f}",
            acc=f"{total_correct / total_count:.4f}",
        )

    gate_weights = np.concatenate(gate_weight_batches, axis=0)
    return (
        total_loss / total_count,
        total_correct / total_count,
        np.array(y_true),
        np.array(y_pred),
        gate_weights,
    )
"""


CELL_33_CODE = """
def train_and_evaluate_gated_seed(seed):
    seed_everything(seed)
    run_slug = f"{FUSION_RUN_SLUG}_seed_{seed}"
    run_dir = OUTPUT_DIR / FUSION_RUN_SLUG / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = run_dir / "best_checkpoint.pt"
    last_checkpoint_path = run_dir / "last_checkpoint.pt"

    train_loader, val_loader, test_loader = build_multimodal_loaders(seed)
    model = SelectedEncoderGatedFusionModel(
        encoder_spec=TEXT_ENCODER_SPEC,
        num_classes=num_classes,
        fusion_block=None,
        fusion_dim=256,
        fusion_dropout=0.2,
        freeze_bert=FREEZE_BERT,
        freeze_image_backbone=FREEZE_IMAGE_BACKBONE,
        cache_dir=HF_CACHE_DIR,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = configure_optimizer(
        model,
        lr_text=LR_TEXT,
        lr_image=LR_IMAGE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

    def move_optimizer_state_to_device(optimizer, device):
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)

    history = []
    best_val_acc = -math.inf
    best_epoch = -1
    start_epoch = 1
    resumed_from_checkpoint = False

    if last_checkpoint_path.exists():
        resume_checkpoint = torch.load(last_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        if "optimizer_state_dict" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            move_optimizer_state_to_device(optimizer, device)
        if "scheduler_state_dict" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        history = list(resume_checkpoint.get("history", []))
        best_val_acc = float(resume_checkpoint.get("best_val_acc", best_val_acc))
        best_epoch = int(resume_checkpoint.get("best_epoch", resume_checkpoint.get("epoch", best_epoch)))
        completed_epoch = int(resume_checkpoint.get("epoch", 0))
        start_epoch = completed_epoch + 1
        resumed_from_checkpoint = True
        print(
            f"Resuming seed {seed} from epoch {completed_epoch}; "
            f"next epoch is {start_epoch}/{EPOCHS}."
        )

    if start_epoch > EPOCHS:
        print(f"Seed {seed} already has a complete last checkpoint at epoch {start_epoch - 1}.")

    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"{SELECTED_ENCODER_NAME} Residual Gated Fusion seed {seed} - Epoch {epoch}/{EPOCHS}")
        train_loss, train_acc = train_epoch_fusion(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        val_loss, val_acc, _, _, _ = evaluate_fusion(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step()
        lr_values = [group["lr"] for group in optimizer.param_groups]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr_min": min(lr_values),
            "lr_max": max(lr_values),
        })

        print(
            f"seed={seed} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )
        print()

        is_best = val_acc >= best_val_acc
        if is_best:
            best_val_acc = float(val_acc)
            best_epoch = epoch

        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "encoder_spec": TEXT_ENCODER_SPEC,
            "seed": seed,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "current_val_acc": float(val_acc),
            "history": history,
            "label_names": label_names,
            "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "epochs": EPOCHS,
            "lr_text": LR_TEXT,
            "lr_image": LR_IMAGE,
            "weight_decay": WEIGHT_DECAY,
            "max_len": MAX_LEN,
            "freeze_bert": FREEZE_BERT,
            "freeze_image_backbone": FREEZE_IMAGE_BACKBONE,
            "lr_schedule_name": LR_SCHEDULE_NAME,
            "fusion_block": "ResidualGatedFusionBlock",
            "checkpoint_policy": "last checkpoint saved every epoch; best checkpoint saved on validation accuracy improvement",
            "comparison_settings": INHERITED_NOTEBOOK02_SETTINGS,
        }
        torch.save(checkpoint_payload, last_checkpoint_path)
        if is_best:
            torch.save(checkpoint_payload, best_checkpoint_path)

    history_df = pd.DataFrame(history)
    history_df.to_csv(run_dir / "training_history.csv", index=False)
    training_curve_path = run_dir / "training_curves.png"
    plot_history(
        history_df,
        f"{SELECTED_ENCODER_NAME} Residual Gated Fusion seed {seed}",
        save_path=training_curve_path,
    )
    print("Saved epoch training curve:", training_curve_path)

    evaluation_checkpoint_path = best_checkpoint_path if best_checkpoint_path.exists() else last_checkpoint_path
    best_checkpoint = torch.load(evaluation_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_loss, test_acc, y_true, y_pred, gate_weights = evaluate_fusion(
        model,
        test_loader,
        criterion,
        device,
    )

    metrics = evaluate_predictions(y_true, y_pred, label_names)
    save_json(metrics, run_dir / "test_metrics.json")
    save_json(metrics["classification_report"], run_dir / "test_classification_report.json")
    pd.DataFrame(metrics["confusion_matrix"], index=label_names, columns=label_names).to_csv(
        run_dir / "test_confusion_matrix.csv"
    )
    save_confusion_matrix_figure(
        metrics["confusion_matrix"],
        f"{SELECTED_ENCODER_NAME} Residual Gated Fusion seed {seed} Test Confusion Matrix",
        run_dir / "test_confusion_matrix.png",
    )

    gate_summary = pd.DataFrame(
        {
            "image_gate_weight": gate_weights[:, 0],
            "text_gate_weight": gate_weights[:, 1],
        }
    )
    gate_summary.to_csv(run_dir / "gate_weights.csv", index=False)

    summary = {
        "model": f"{SELECTED_ENCODER_NAME} residual gated fusion",
        "source": "notebook_03",
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(metrics["macro_f1"]),
        "checkpoint_path": str(evaluation_checkpoint_path),
        "best_checkpoint_path": str(best_checkpoint_path),
        "last_checkpoint_path": str(last_checkpoint_path),
        "training_curve_path": str(training_curve_path),
        "checkpoint_policy": "best checkpoint tracks validation accuracy; last checkpoint updates every epoch",
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "resume_start_epoch": start_epoch,
        "run_dir": str(run_dir),
    }
    save_json(summary, run_dir / "run_summary.json")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


gated_seed_summaries = []
for seed in EXPERIMENT_SEEDS:
    gated_seed_summaries.append(train_and_evaluate_gated_seed(seed))

gated_seed_summary_df = pd.DataFrame(gated_seed_summaries)
gated_seed_summary_df.to_csv(OUTPUT_DIR / FUSION_RUN_SLUG / "gated_fusion_seed_summary.csv", index=False)
display(gated_seed_summary_df)
"""


CELL_34_CODE = """
gated_aggregate_row = {
    "model": f"{SELECTED_ENCODER_NAME} residual gated fusion",
    "source": "notebook_03",
    "test_accuracy_mean": float(gated_seed_summary_df["test_accuracy"].mean()),
    "test_macro_f1_mean": float(gated_seed_summary_df["test_macro_f1"].mean()),
    "val_accuracy_mean": float(gated_seed_summary_df["best_val_acc"].mean()),
    "val_macro_f1_mean": np.nan,
}

gated_aggregate_df = pd.DataFrame([gated_aggregate_row])
gated_aggregate_df.to_csv(OUTPUT_DIR / FUSION_RUN_SLUG / "gated_fusion_aggregate_summary.csv", index=False)
display(gated_aggregate_df)
"""


CELL_36_CODE = """
comparison_df = pd.concat(
    [notebook02_baselines_df, gated_aggregate_df],
    ignore_index=True,
)
comparison_df = comparison_df.sort_values(
    ["test_accuracy_mean", "test_macro_f1_mean"],
    ascending=[False, False],
).reset_index(drop=True)
comparison_df = comparison_df.drop(columns=["seed"], errors="ignore")

comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
display(comparison_df)
"""


CELL_35_MARKDOWN = """
## Compare model performance
"""


CELL_37_MARKDOWN = """
## Conclusion

The Bio_ClinicalBERT residual gated fusion model is the strongest test-set result in this comparison. It improves over the Notebook 02 standard Bio_ClinicalBERT fusion baseline and the dual-view image baseline, while the text-only baseline remains much weaker.

The main caveat is that the gated model does not need to dominate every validation metric to be useful; the decision to carry it forward is based on the stronger held-out test accuracy and macro F1. Later notebooks should still treat it as a promising architecture result rather than a final model choice.

The run saves a best validation checkpoint, a last-epoch recovery checkpoint, the epoch-level training curve, test metrics, classification reports, confusion matrices, and gate weights. The checkpoints make the selected model recoverable, while the training curve remains important for judging overfitting and loss behavior.
"""


# Clear execution counts and outputs before writing the notebook.
def clear_execution_state(nb: dict) -> None:
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []


# Insert or replace one notebook cell while keeping notebook structure valid.
def set_cell(nb: dict, index: int, source: str, cell_type: str) -> None:
    while len(nb["cells"]) <= index:
        nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": []})
    cell = nb["cells"][index]
    cell["cell_type"] = cell_type
    cell["source"] = to_source(source)
    cell.setdefault("id", cell_id(index, cell_type, source))
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    else:
        cell.pop("execution_count", None)
        cell.pop("outputs", None)


# Apply the Experiment 03 cell updates and trim obsolete trailing cells.
def update_cells(nb: dict) -> None:
    set_cell(nb, 0, CELL_0_MARKDOWN, "markdown")
    set_cell(nb, 2, CELL_2_CODE, "code")
    set_cell(nb, 6, CELL_6_CODE, "code")
    set_cell(nb, 13, CELL_13_MARKDOWN, "markdown")
    set_cell(nb, 14, CELL_14_CODE, "code")
    set_cell(nb, 18, CELL_18_CODE, "code")
    set_cell(nb, 20, CELL_20_CODE, "code")
    set_cell(nb, 22, CELL_22_CODE, "code")
    set_cell(nb, 23, CELL_23_MARKDOWN, "markdown")
    set_cell(nb, 24, CELL_24_CODE, "code")
    set_cell(nb, 25, CELL_25_CODE, "code")
    set_cell(nb, 26, CELL_27_MARKDOWN, "markdown")
    set_cell(nb, 27, CELL_28_CODE, "code")
    set_cell(nb, 28, CELL_29_CODE, "code")
    set_cell(nb, 29, CELL_30_CODE, "code")
    set_cell(nb, 30, CELL_30_CLEANUP_CODE, "code")
    set_cell(nb, 31, CELL_31_MARKDOWN, "markdown")
    set_cell(nb, 32, CELL_32_CODE, "code")
    set_cell(nb, 33, CELL_33_CODE, "code")
    set_cell(nb, 34, CELL_34_CODE, "code")
    set_cell(nb, 35, CELL_35_MARKDOWN, "markdown")
    set_cell(nb, 36, CELL_36_CODE, "code")
    set_cell(nb, 37, CELL_37_MARKDOWN, "markdown")
    nb["cells"] = nb["cells"][:38]


# Generate the gated-fusion notebook from the selected source notebook.
def main() -> None:
    source_notebook = resolve_source_notebook()
    nb = json.loads(source_notebook.read_text(encoding="utf-8"))
    clear_execution_state(nb)
    update_cells(nb)
    TARGET_NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Used source notebook: {source_notebook}")
    print(f"Wrote {TARGET_NOTEBOOK}")


if __name__ == "__main__":
    main()
