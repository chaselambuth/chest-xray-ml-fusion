# Build the Experiment 02 text-encoder benchmark notebook from cell templates.

from __future__ import annotations

import json
import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_NOTEBOOK = PROJECT_ROOT / "Experiments" / "02_Xray_Text_Encoder_Selection.ipynb"
SOURCE_NOTEBOOK_CANDIDATES = [
    PROJECT_ROOT / "Experiments" / "02_Xray_Text_Encoder_Selection.ipynb",
    PROJECT_ROOT / "Experiments" / "02_Xray_ClinicalBERT_Fusion.ipynb",
    PROJECT_ROOT / "Experiments" / "02_Outdated_Notebook.ipynb",
    PROJECT_ROOT / "Experiments" / "archive" / "02_Outdated_Notebook.ipynb",
]


# Pick the best available prior notebook to preserve metadata from.
def resolve_source_notebook() -> Path:
    for candidate in SOURCE_NOTEBOOK_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find an Experiment 02 source notebook. Checked: "
        + ", ".join(str(path) for path in SOURCE_NOTEBOOK_CANDIDATES)
    )


# Convert a multiline string into notebook source-line format.
def to_source(text: str) -> list[str]:
    text = text.strip("\n") + "\n"
    return list(text.splitlines(keepends=True))


# Create a stable short cell id from cell type and source text.
def cell_id(cell_type: str, text: str) -> str:
    digest = hashlib.sha1(f"{cell_type}:{text}".encode("utf-8")).hexdigest()
    return digest[:12]


# Build a markdown cell dictionary for nbformat JSON.
def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id("markdown", text),
        "metadata": {},
        "source": to_source(text),
    }


# Build an unexecuted code cell dictionary for nbformat JSON.
def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "id": cell_id("code", text),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": to_source(text),
    }


CELL_0_MARKDOWN = """
# Experiment 2: Text Encoder Selection Benchmark for Dual-View Chest X-ray + Multimodal Fusion

## Goal

This experiment reuses the stronger `02_Xray_ClinicalBERT_Fusion.ipynb` setup and keeps the following unchanged across runs:

- dataset splits
- image preprocessing
- dual-view ResNet18 image encoder
- concatenation-based fusion architecture
- max token length
- batch size
- optimizer family
- learning rate schedule
- random seed list
- number of training epochs

The primary experimental variable is the text backbone, while a shared dual-image baseline is also retained so this notebook fully covers the comparison made in `02`.

## Text Backbones Compared

1. Bio_ClinicalBERT
2. CheXbert
3. RadBERT

Bio_ClinicalBERT and RadBERT are loaded through Hugging Face `AutoModel`.
CheXbert is loaded from the Stanford AIMI scorer checkpoint and its BERT encoder weights are reused as the text backbone.

## Outputs

This notebook trains:

- one shared dual-image classifier baseline
- a text-only classifier
- a fused image + text classifier

Artifacts are saved under the existing `Experiments/outputs/experiment_02_text_encoder_benchmark/` benchmark layout, including:

- best checkpoint for each encoder and task
- training history
- validation and test metrics, including macro AUROC and macro AUPRC when computable
- per-class precision/recall tables
- per-class AUROC/AUPRC tables when computable
- confusion matrices
- per-seed and aggregate CSV/JSON summaries across all runs
"""


CELL_1_MARKDOWN = """
## Imports and configuration
"""


CELL_2_CODE = """
from pathlib import Path
import gc
import json
import math
import os
import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, label_binarize

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel
from transformers.utils.hub import cached_file

BASE_SEED = 42
EXPERIMENT_SEEDS = [42]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(BASE_SEED)

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
EPOCHS = 3
LR_TEXT = 2e-5
LR_IMAGE = 1e-4
WEIGHT_DECAY = 1e-4
MAX_LEN = 64
FREEZE_BERT = True
FREEZE_IMAGE_BACKBONE = False

LR_SCHEDULE_NAME = "constant"
EXPERIMENT_NAME = "experiment_02_text_encoder_benchmark"
OUTPUT_DIR = base_dir / "Experiments" / "outputs" / EXPERIMENT_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESOLVED_MODEL_CACHE_DIR = OUTPUT_DIR / "resolved_hf_models"
RESOLVED_MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
print("Saving artifacts to:", OUTPUT_DIR)

TEXT_ENCODER_SPECS = [
    {
        "name": "Bio_ClinicalBERT",
        "slug": "bio_clinicalbert",
        "loader": "auto",
        "model_name": "emilyalsentzer/Bio_ClinicalBERT",
        "tokenizer_name": "emilyalsentzer/Bio_ClinicalBERT",
        "tokenizer_note": "Uses the tokenizer shipped with the Bio_ClinicalBERT checkpoint.",
        "description": "Clinical notes model initialized from BioBERT and trained on MIMIC notes.",
    },
    {
        "name": "CheXbert",
        "slug": "chexbert",
        "loader": "chexbert",
        "repo_id": "StanfordAIMI/RRG_scorers",
        "filename": "chexbert.pth",
        "tokenizer_name": "bert-base-uncased",
        "tokenizer_note": (
            "The public CheXbert artifact provides BERT weights but no tokenizer files. "
            "The loader constructs BertModel(BertConfig()), whose default 30,522-token "
            "vocabulary matches the BERT-base uncased tokenizer, so this run uses "
            "bert-base-uncased instead of reusing Bio_ClinicalBERT."
        ),
        "description": "Radiology-report BERT checkpoint used by Stanford AIMI CheXbert scoring code.",
    },
    {
        "name": "RadBERT",
        "slug": "radbert",
        "loader": "auto",
        "model_name": "StanfordAIMI/RadBERT",
        "tokenizer_name": "StanfordAIMI/RadBERT",
        "tokenizer_note": "Uses the tokenizer shipped with the RadBERT checkpoint.",
        "description": "Radiology-tuned transformer continuously pretrained on radiology reports.",
    },
]

print("Experiment seeds:", EXPERIMENT_SEEDS)
print("Text encoders configured:")
for spec in TEXT_ENCODER_SPECS:
    print(f" - {spec['name']} | tokenizer: {spec['tokenizer_name']}")
"""


CELL_3_MARKDOWN = """
## Load precomputed train, validation, and test splits
"""


CELL_4_CODE = """
train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)
test_df = pd.read_csv(test_csv)

print("Train shape:", train_df.shape)
print("Val shape:", val_df.shape)
print("Test shape:", test_df.shape)
display(train_df.head())
"""


CELL_5_MARKDOWN = """
## Encode labels and build image transforms
"""


CELL_6_CODE = """
label_encoder = LabelEncoder()
train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["label_enc"] = label_encoder.fit_transform(train_df["clean_label"])
val_df["label_enc"] = label_encoder.transform(val_df["clean_label"])
test_df["label_enc"] = label_encoder.transform(test_df["clean_label"])

label_names = list(label_encoder.classes_)
num_classes = len(label_names)

print("Classes:", label_names)

image_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])
"""


CELL_7_MARKDOWN = """
## Dataset classes and dataloaders
"""


CELL_8_CODE = """
class ClinicalTextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=MAX_LEN):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["indication"]) if pd.notna(row["indication"]) else ""

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(row["label_enc"]), dtype=torch.long),
        }


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
            "frontal": frontal,
            "lateral": lateral,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(row["label_enc"]), dtype=torch.long),
        }


class XRayDualImageDataset(Dataset):
    def __init__(self, df, image_transform=None):
        self.df = df.reset_index(drop=True)
        self.image_transform = image_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        frontal = Image.open(row["frontal_path"]).convert("RGB")
        lateral = Image.open(row["lateral_path"]).convert("RGB")

        if self.image_transform is not None:
            frontal = self.image_transform(frontal)
            lateral = self.image_transform(lateral)

        return {
            "frontal": frontal,
            "lateral": lateral,
            "labels": torch.tensor(int(row["label_enc"]), dtype=torch.long),
        }


def make_torch_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def make_loader(dataset, seed, shuffle):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        generator=make_torch_generator(seed) if shuffle else None,
    )


def load_tokenizer_for_encoder(encoder_spec):
    tokenizer_name = encoder_spec["tokenizer_name"]
    local_source = resolve_tokenizer_source(tokenizer_name)
    tokenizer_source = local_source if local_source is not None else tokenizer_name
    local_only = local_source is not None
    print(f"Loading tokenizer for {encoder_spec['name']}: {tokenizer_source}")
    print("Tokenizer note:", encoder_spec["tokenizer_note"])
    return AutoTokenizer.from_pretrained(
        tokenizer_source,
        cache_dir=HF_CACHE_DIR,
        local_files_only=local_only,
    )


def build_text_loaders(tokenizer, seed):
    train_ds = ClinicalTextDataset(train_df, tokenizer)
    val_ds = ClinicalTextDataset(val_df, tokenizer)
    test_ds = ClinicalTextDataset(test_df, tokenizer)
    return (
        make_loader(train_ds, seed=seed, shuffle=True),
        make_loader(val_ds, seed=seed, shuffle=False),
        make_loader(test_ds, seed=seed, shuffle=False),
    )


def build_multimodal_loaders(tokenizer, seed):
    train_ds = XRayMultimodalDataset(train_df, tokenizer, image_transform=image_transform)
    val_ds = XRayMultimodalDataset(val_df, tokenizer, image_transform=image_transform)
    test_ds = XRayMultimodalDataset(test_df, tokenizer, image_transform=image_transform)
    return (
        make_loader(train_ds, seed=seed, shuffle=True),
        make_loader(val_ds, seed=seed, shuffle=False),
        make_loader(test_ds, seed=seed, shuffle=False),
    )


def build_image_loaders(seed):
    train_ds = XRayDualImageDataset(train_df, image_transform=image_transform)
    val_ds = XRayDualImageDataset(val_df, image_transform=image_transform)
    test_ds = XRayDualImageDataset(test_df, image_transform=image_transform)
    return (
        make_loader(train_ds, seed=seed, shuffle=True),
        make_loader(val_ds, seed=seed, shuffle=False),
        make_loader(test_ds, seed=seed, shuffle=False),
    )

class_counts = train_df["clean_label"].value_counts()
class_weights = torch.tensor(
    [len(train_df) / class_counts[label] for label in label_names],
    dtype=torch.float32,
    device=device,
)

sample_img_loader, _, _ = build_image_loaders(BASE_SEED)
sample_batch = next(iter(sample_img_loader))
for key, value in sample_batch.items():
    print(key, tuple(value.shape))
"""


CELL_9_MARKDOWN = """
## Model definitions, training utilities, and artifact helpers
"""


CELL_10_CODE = """
def get_cls_embedding(outputs):
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state[:, 0, :]
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    raise ValueError("Unsupported transformer output: unable to extract CLS embedding.")


def hf_repo_cache_dir(repo_id, cache_dir=HF_CACHE_DIR):
    return Path(cache_dir) / f"models--{repo_id.replace('/', '--')}"


def iter_hf_snapshots(repo_id, cache_dir=HF_CACHE_DIR):
    snapshots_dir = hf_repo_cache_dir(repo_id, cache_dir) / "snapshots"
    if not snapshots_dir.exists():
        return []
    return sorted([path for path in snapshots_dir.iterdir() if path.is_dir()])


def first_cached_file(repo_id, filenames, cache_dir=HF_CACHE_DIR):
    for filename in filenames:
        for snapshot_dir in iter_hf_snapshots(repo_id, cache_dir):
            candidate = snapshot_dir / filename
            if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
    return None


def copy_if_needed(source_path, target_path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and target_path.stat().st_size == source_path.stat().st_size:
        return
    shutil.copy2(source_path, target_path)


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
    missing, unexpected = bert.load_state_dict(bert_state, strict=False)
    print("CheXbert backbone loaded.")
    if missing:
        print("Missing CheXbert keys:", missing[:10])
    if unexpected:
        print("Unexpected CheXbert keys:", unexpected[:10])
    return bert


def load_text_backbone(spec, cache_dir=None):
    if spec["loader"] == "auto":
        backbone = load_auto_backbone(spec["model_name"], cache_dir=cache_dir)
    elif spec["loader"] == "chexbert":
        backbone = load_chexbert_backbone(
            spec["repo_id"],
            spec["filename"],
            cache_dir=cache_dir,
        )
    else:
        raise ValueError(f"Unknown loader type: {spec['loader']}")

    hidden_size = backbone.config.hidden_size
    return backbone, hidden_size


class TextOnlyClassifier(nn.Module):
    def __init__(self, encoder_spec, num_classes, freeze_bert=False, cache_dir=None):
        super().__init__()
        self.encoder_spec = encoder_spec
        self.bert, hidden_size = load_text_backbone(encoder_spec, cache_dir=cache_dir)

        if freeze_bert:
            print(f"Freezing text encoder for {encoder_spec['name']} text-only model...")
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
        text_features = self.encode_text(input_ids, attention_mask)
        return self.classifier(text_features)


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
"""


CELL_10_CODE += """


class DualImageClassifier(nn.Module):
    def __init__(self, num_classes, freeze_image_backbone=False):
        super().__init__()
        self.image_encoder = DualImageEncoder(freeze_backbone=freeze_image_backbone)
        self.classifier = nn.Sequential(
            nn.Linear(self.image_encoder.output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, frontal, lateral):
        image_features = self.image_encoder(frontal, lateral)
        return self.classifier(image_features)


class FusionClassifier(nn.Module):
    def __init__(
        self,
        encoder_spec,
        num_classes,
        freeze_bert=False,
        freeze_image_backbone=False,
        cache_dir=None,
    ):
        super().__init__()
        self.encoder_spec = encoder_spec
        self.image_encoder = DualImageEncoder(freeze_backbone=freeze_image_backbone)
        self.bert, hidden_size = load_text_backbone(encoder_spec, cache_dir=cache_dir)

        if freeze_bert:
            print(f"Freezing text encoder for {encoder_spec['name']} fusion model...")
            for param in self.bert.parameters():
                param.requires_grad = False

        self.text_proj = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        fusion_dim = self.image_encoder.output_dim + 256
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def encode_text(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = get_cls_embedding(outputs)
        return self.text_proj(cls_embedding)

    def forward(self, frontal, lateral, input_ids, attention_mask):
        image_features = self.image_encoder(frontal, lateral)
        text_features = self.encode_text(input_ids, attention_mask)
        fused_features = torch.cat([image_features, text_features], dim=1)
        return self.classifier(fused_features)


def configure_optimizer(model, lr_text=LR_TEXT, lr_image=LR_IMAGE, weight_decay=WEIGHT_DECAY):
    bert_params = []
    image_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "bert" in name:
            bert_params.append(param)
        elif "image_encoder" in name or "backbone" in name:
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
    return optimizer


def build_scheduler(optimizer):
    return LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)


def move_batch_to_device(batch, task_type, device):
    if task_type == "image_only":
        return {
            "frontal": batch["frontal"].to(device),
            "lateral": batch["lateral"].to(device),
            "labels": batch["labels"].to(device),
        }

    if task_type == "text_only":
        return {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "labels": batch["labels"].to(device),
        }

    if task_type == "fusion":
        return {
            "frontal": batch["frontal"].to(device),
            "lateral": batch["lateral"].to(device),
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "labels": batch["labels"].to(device),
        }

    raise ValueError(f"Unknown task_type: {task_type}")


def forward_task(model, batch, task_type):
    if task_type == "image_only":
        return model(batch["frontal"], batch["lateral"])
    if task_type == "text_only":
        return model(batch["input_ids"], batch["attention_mask"])
    if task_type == "fusion":
        return model(
            batch["frontal"],
            batch["lateral"],
            batch["input_ids"],
            batch["attention_mask"],
        )
    raise ValueError(f"Unknown task_type: {task_type}")


def train_one_epoch(model, loader, criterion, optimizer, device, task_type, desc):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    progress = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for batch in progress:
        batch = move_batch_to_device(batch, task_type, device)
        labels = batch["labels"]

        optimizer.zero_grad()
        logits = forward_task(model, batch, task_type)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_count += batch_size

        progress.set_postfix(
            loss=f"{total_loss / total_count:.4f}",
            acc=f"{total_correct / total_count:.4f}",
        )

    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def evaluate_model(model, loader, criterion, device, task_type, desc):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    y_true = []
    y_pred = []
    y_prob = []

    progress = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for batch in progress:
        batch = move_batch_to_device(batch, task_type, device)
        labels = batch["labels"]

        logits = forward_task(model, batch, task_type)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_count += batch_size

        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        y_prob.extend(probs.detach().cpu().tolist())

        progress.set_postfix(
            loss=f"{total_loss / total_count:.4f}",
            acc=f"{total_correct / total_count:.4f}",
        )

    return (
        total_loss / total_count,
        total_correct / total_count,
        np.array(y_true),
        np.array(y_pred),
        np.array(y_prob),
    )
"""


CELL_10_CODE += """


def safe_metric(metric_fn, *args, **kwargs):
    try:
        value = metric_fn(*args, **kwargs)
    except ValueError:
        return np.nan
    return float(value)


def compute_probability_metrics(y_true, y_prob, label_names):
    class_ids = np.arange(len(label_names))
    y_true_bin = label_binarize(y_true, classes=class_ids)
    if len(label_names) == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.column_stack([1 - y_true_bin[:, 0], y_true_bin[:, 0]])

    per_class_rows = []
    for class_idx, class_name in enumerate(label_names):
        class_true = y_true_bin[:, class_idx]
        class_prob = y_prob[:, class_idx]
        per_class_rows.append(
            {
                "class_name": class_name,
                "auroc": safe_metric(roc_auc_score, class_true, class_prob),
                "auprc": safe_metric(average_precision_score, class_true, class_prob),
            }
        )

    per_class_prob_df = pd.DataFrame(per_class_rows)
    macro_auroc = float(per_class_prob_df["auroc"].mean(skipna=True))
    macro_auprc = float(per_class_prob_df["auprc"].mean(skipna=True))

    return {
        "macro_auroc": macro_auroc,
        "macro_auprc": macro_auprc,
        "per_class_probability_df": per_class_prob_df,
    }


def compute_metrics(y_true, y_pred, y_prob, label_names):
    report = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)
    per_class_df = (
        pd.DataFrame(report)
        .transpose()
        .iloc[: len(label_names)]
        .reset_index()
        .rename(columns={"index": "class_name"})
    )
    probability_metrics = compute_probability_metrics(y_true, y_prob, label_names)
    per_class_df = per_class_df.merge(
        probability_metrics["per_class_probability_df"],
        on="class_name",
        how="left",
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "macro_auroc": probability_metrics["macro_auroc"],
        "macro_auprc": probability_metrics["macro_auprc"],
        "report": report,
        "per_class_df": per_class_df,
        "confusion_matrix": cm,
    }


def save_json(payload, path):
    def to_jsonable(value):
        if isinstance(value, dict):
            return {key: to_jsonable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [to_jsonable(item) for item in value]
        if isinstance(value, tuple):
            return [to_jsonable(item) for item in value]
        if isinstance(value, np.integer):
            return value.item()
        if isinstance(value, np.floating):
            value = value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, float) and np.isnan(value):
            return None
        return value

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, allow_nan=False)


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

    for row_idx in range(len(label_names)):
        for col_idx in range(len(label_names)):
            ax.text(col_idx, row_idx, int(cm[row_idx, col_idx]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_history(history_df, title, save_path):
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
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def checkpoint_payload(model, optimizer, scheduler, encoder_spec, task_type, seed, epoch, best_val_acc):
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "encoder_spec": encoder_spec,
        "task_type": task_type,
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "seed": seed,
        "label_names": label_names,
        "max_len": MAX_LEN,
        "batch_size": BATCH_SIZE,
        "lr_schedule_name": LR_SCHEDULE_NAME,
    }


def save_split_artifacts(split_name, metrics, run_dir, title_prefix):
    metrics_summary = {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "macro_auroc": metrics["macro_auroc"],
        "macro_auprc": metrics["macro_auprc"],
    }
    save_json(metrics_summary, run_dir / f"{split_name}_metrics.json")
    save_json(metrics["report"], run_dir / f"{split_name}_classification_report.json")

    per_class_path = run_dir / f"{split_name}_per_class_metrics.csv"
    cm_csv_path = run_dir / f"{split_name}_confusion_matrix.csv"
    cm_png_path = run_dir / f"{split_name}_confusion_matrix.png"

    metrics["per_class_df"].to_csv(per_class_path, index=False)
    pd.DataFrame(
        metrics["confusion_matrix"],
        index=label_names,
        columns=label_names,
    ).to_csv(cm_csv_path)
    save_confusion_matrix_figure(
        metrics["confusion_matrix"],
        f"{title_prefix} {split_name.title()} Confusion Matrix",
        cm_png_path,
    )


def extract_per_class_rows(metrics, encoder_name, encoder_slug, task_type, seed, split_name):
    df = metrics["per_class_df"].copy()
    df.insert(0, "split", split_name)
    df.insert(0, "seed", seed)
    df.insert(0, "task_type", task_type)
    df.insert(0, "encoder_slug", encoder_slug)
    df.insert(0, "encoder_name", encoder_name)
    return df
"""


CELL_10_CODE += """


def train_and_evaluate_run(encoder_spec, task_type, seed):
    seed_everything(seed)

    if task_type == "image_only":
        encoder_name = "DualImageResNet18"
        encoder_slug = "dual_image_resnet18"
        run_slug = "dual_image_baseline"
        tokenizer_name = None
        tokenizer_note = "Image-only baseline does not use text tokenization."
    else:
        encoder_name = encoder_spec["name"]
        encoder_slug = encoder_spec["slug"]
        run_slug = f"{encoder_slug}_{task_type}"
        tokenizer_name = encoder_spec["tokenizer_name"]
        tokenizer_note = encoder_spec["tokenizer_note"]

    run_dir = OUTPUT_DIR / run_slug / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best_checkpoint.pt"

    if task_type == "image_only":
        model = DualImageClassifier(
            num_classes=num_classes,
            freeze_image_backbone=FREEZE_IMAGE_BACKBONE,
        ).to(device)
        train_loader, val_loader, test_loader = build_image_loaders(seed)
        run_title = "Dual-Image ResNet18 Baseline"
    elif task_type == "text_only":
        tokenizer = load_tokenizer_for_encoder(encoder_spec)
        model = TextOnlyClassifier(
            encoder_spec=encoder_spec,
            num_classes=num_classes,
            freeze_bert=FREEZE_BERT,
            cache_dir=HF_CACHE_DIR,
        ).to(device)
        train_loader, val_loader, test_loader = build_text_loaders(tokenizer, seed)
        run_title = f"{encoder_name} Text-Only"
    elif task_type == "fusion":
        tokenizer = load_tokenizer_for_encoder(encoder_spec)
        model = FusionClassifier(
            encoder_spec=encoder_spec,
            num_classes=num_classes,
            freeze_bert=FREEZE_BERT,
            freeze_image_backbone=FREEZE_IMAGE_BACKBONE,
            cache_dir=HF_CACHE_DIR,
        ).to(device)
        train_loader, val_loader, test_loader = build_multimodal_loaders(tokenizer, seed)
        run_title = f"{encoder_name} Fusion"
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = configure_optimizer(model)
    scheduler = build_scheduler(optimizer)

    history = []
    best_val_acc = -math.inf
    best_epoch = -1

    for epoch in range(1, EPOCHS + 1):
        print(f"{run_title} - Epoch {epoch}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            task_type=task_type,
            desc=f"Train {run_slug}",
        )

        val_loss, val_acc, _, _, _ = evaluate_model(
            model,
            val_loader,
            criterion,
            device,
            task_type=task_type,
            desc=f"Val {run_slug}",
        )

        scheduler.step()

        lr_values = [group["lr"] for group in optimizer.param_groups]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr_min": min(lr_values),
                "lr_max": max(lr_values),
            }
        )

        print(
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )
        print()

        if val_acc >= best_val_acc:
            best_val_acc = float(val_acc)
            best_epoch = epoch
            torch.save(
                checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    encoder_spec,
                    task_type,
                    seed,
                    epoch,
                    best_val_acc,
                ),
                checkpoint_path,
            )

    history_df = pd.DataFrame(history)
    history_df.to_csv(run_dir / "training_history.csv", index=False)
    plot_history(history_df, run_title, run_dir / "training_curves.png")

    best_checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    val_loss, val_acc, val_y_true, val_y_pred, val_y_prob = evaluate_model(
        model,
        val_loader,
        criterion,
        device,
        task_type=task_type,
        desc=f"Best Val {run_slug}",
    )
    test_loss, test_acc, test_y_true, test_y_pred, test_y_prob = evaluate_model(
        model,
        test_loader,
        criterion,
        device,
        task_type=task_type,
        desc=f"Test {run_slug}",
    )

    val_metrics = compute_metrics(val_y_true, val_y_pred, val_y_prob, label_names)
    test_metrics = compute_metrics(test_y_true, test_y_pred, test_y_prob, label_names)

    save_split_artifacts("validation", val_metrics, run_dir, run_title)
    save_split_artifacts("test", test_metrics, run_dir, run_title)

    run_summary = {
        "encoder_name": encoder_name,
        "encoder_slug": encoder_slug,
        "task_type": task_type,
        "seed": seed,
        "tokenizer_name": tokenizer_name,
        "tokenizer_note": tokenizer_note,
        "best_epoch": best_epoch,
        "val_loss": float(val_loss),
        "val_accuracy": float(val_acc),
        "val_macro_f1": float(val_metrics["macro_f1"]),
        "val_macro_auroc": float(val_metrics["macro_auroc"]),
        "val_macro_auprc": float(val_metrics["macro_auprc"]),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_macro_auroc": float(test_metrics["macro_auroc"]),
        "test_macro_auprc": float(test_metrics["macro_auprc"]),
        "checkpoint_path": str(checkpoint_path),
        "run_dir": str(run_dir),
    }
    save_json(run_summary, run_dir / "run_summary.json")

    per_class_frames = [
        extract_per_class_rows(val_metrics, encoder_name, encoder_slug, task_type, seed, "validation"),
        extract_per_class_rows(test_metrics, encoder_name, encoder_slug, task_type, seed, "test"),
    ]
    run_per_class_df = pd.concat(per_class_frames, ignore_index=True)
    run_per_class_df.to_csv(run_dir / "per_class_metrics_all_splits.csv", index=False)

    display(history_df.tail(1))
    display(run_per_class_df)

    print(f"{run_title} validation accuracy:", round(run_summary["val_accuracy"], 4))
    print(f"{run_title} test accuracy:", round(run_summary["test_accuracy"], 4))
    print(f"{run_title} test macro F1:", round(run_summary["test_macro_f1"], 4))
    print(f"{run_title} test macro AUROC:", round(run_summary["test_macro_auroc"], 4))
    print(f"{run_title} test macro AUPRC:", round(run_summary["test_macro_auprc"], 4))
    print("Best checkpoint:", checkpoint_path)
    print()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return run_summary, run_per_class_df
"""


CELL_11_MARKDOWN = """
## Run the text encoder benchmark
"""


CELL_12_CODE = """
run_summaries = []
per_class_tables = []
skipped_runs = []

for seed in EXPERIMENT_SEEDS:
    try:
        image_summary, image_per_class_df = train_and_evaluate_run(None, "image_only", seed)
        run_summaries.append(image_summary)
        per_class_tables.append(image_per_class_df)
    except Exception as exc:
        skip_record = {
            "encoder_name": "DualImageResNet18",
            "encoder_slug": "dual_image_resnet18",
            "task_type": "image_only",
            "seed": seed,
            "tokenizer_name": None,
            "status": "skipped",
            "error": str(exc),
        }
        skipped_runs.append(skip_record)
        print(f"Skipping dual-image baseline seed {seed} due to error: {exc}")
        print()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

for encoder_spec in TEXT_ENCODER_SPECS:
    print("=" * 80)
    print(f"Starting runs for {encoder_spec['name']}")
    print(encoder_spec["description"])
    print("=" * 80)

    for task_type in ["text_only", "fusion"]:
        for seed in EXPERIMENT_SEEDS:
            try:
                run_summary, run_per_class_df = train_and_evaluate_run(encoder_spec, task_type, seed)
                run_summaries.append(run_summary)
                per_class_tables.append(run_per_class_df)
            except Exception as exc:
                skip_record = {
                    "encoder_name": encoder_spec["name"],
                    "encoder_slug": encoder_spec["slug"],
                    "task_type": task_type,
                    "seed": seed,
                    "tokenizer_name": encoder_spec["tokenizer_name"],
                    "status": "skipped",
                    "error": str(exc),
                }
                skipped_runs.append(skip_record)
                print(f"Skipping {encoder_spec['name']} / {task_type} / seed {seed} due to error: {exc}")
                print()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

run_summary_df = pd.DataFrame(run_summaries)
if not run_summary_df.empty:
    task_order = {"image_only": 0, "text_only": 1, "fusion": 2}
    run_summary_df["task_rank"] = run_summary_df["task_type"].map(task_order).fillna(99)
    run_summary_df = run_summary_df.sort_values(
        ["task_rank", "encoder_name", "seed"],
        ascending=[True, True, True],
    ).drop(columns=["task_rank"]).reset_index(drop=True)
    run_summary_df.to_csv(OUTPUT_DIR / "run_summary.csv", index=False)

    aggregate_rows = []
    group_cols = ["encoder_name", "encoder_slug", "task_type", "tokenizer_name"]
    metric_cols = ["test_accuracy", "test_macro_f1", "test_macro_auroc", "test_macro_auprc"]
    for group_key, group_df in run_summary_df.groupby(group_cols, dropna=False):
        group_df = group_df.sort_values("seed")
        row = dict(zip(group_cols, group_key))
        row["n_seeds"] = int(group_df["seed"].nunique())
        row["seeds"] = ",".join(str(seed) for seed in group_df["seed"].tolist())
        for metric_col in metric_cols:
            values = group_df[["seed", metric_col]].dropna()
            metric_values = values[metric_col].astype(float)
            row[f"{metric_col}_mean"] = float(metric_values.mean()) if not metric_values.empty else np.nan
            row[f"{metric_col}_std"] = float(metric_values.std(ddof=1)) if len(metric_values) > 1 else 0.0
            row[f"{metric_col}_per_seed"] = json.dumps(
                {str(int(seed)): float(value) for seed, value in values.to_numpy()},
                sort_keys=True,
            )
        aggregate_rows.append(row)

    aggregate_summary_df = pd.DataFrame(aggregate_rows)
    aggregate_summary_df = aggregate_summary_df.sort_values(
        ["task_type", "test_accuracy_mean", "test_macro_f1_mean"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    aggregate_summary_df.to_csv(OUTPUT_DIR / "aggregate_summary.csv", index=False)
    save_json(aggregate_summary_df.to_dict(orient="records"), OUTPUT_DIR / "aggregate_summary.json")
else:
    aggregate_summary_df = pd.DataFrame()

if per_class_tables:
    per_class_summary_df = pd.concat(per_class_tables, ignore_index=True)
    per_class_summary_df.to_csv(OUTPUT_DIR / "per_class_summary.csv", index=False)
else:
    per_class_summary_df = pd.DataFrame()

skipped_columns = [
    "encoder_name",
    "encoder_slug",
    "task_type",
    "seed",
    "tokenizer_name",
    "status",
    "error",
]
skipped_runs_df = pd.DataFrame(skipped_runs, columns=skipped_columns)
skipped_runs_df.to_csv(OUTPUT_DIR / "skipped_runs.csv", index=False)

print("Benchmark complete.")
print("Run summary CSV:", OUTPUT_DIR / "run_summary.csv")
if not aggregate_summary_df.empty:
    print("Aggregate summary CSV:", OUTPUT_DIR / "aggregate_summary.csv")
if not per_class_summary_df.empty:
    print("Per-class summary CSV:", OUTPUT_DIR / "per_class_summary.csv")
print("Skipped runs CSV:", OUTPUT_DIR / "skipped_runs.csv")
"""


CELL_13_MARKDOWN = """
## Review the exported summaries
"""


CELL_14_CODE = """
if 'run_summary_df' in globals() and not run_summary_df.empty:
    display(run_summary_df)
else:
    print("No successful runs were recorded.")

if 'aggregate_summary_df' in globals() and not aggregate_summary_df.empty:
    display(aggregate_summary_df)

if 'per_class_summary_df' in globals() and not per_class_summary_df.empty:
    display(per_class_summary_df.head(12))

if 'skipped_runs_df' in globals() and not skipped_runs_df.empty:
    display(skipped_runs_df)
"""


CELL_15_MARKDOWN = """
## Notes

- This notebook preserves the `02` experiment's split files, image preprocessing, ResNet18 image encoder, and concatenation-based fusion block while also retaining the dual-image-only baseline from `02`.
- Tokenization is now selected per encoder: Bio_ClinicalBERT uses `emilyalsentzer/Bio_ClinicalBERT`, RadBERT uses `StanfordAIMI/RadBERT`, and CheXbert uses `bert-base-uncased` because the public CheXbert scorer artifact provides BERT weights but no tokenizer files while the loader constructs a default `BertConfig()` backbone.
- Hugging Face assets are loaded from materialized local cache directories when possible. This handles partially split local cache snapshots, such as a config in one snapshot and model weights in another, before falling back to online Hub downloads for genuinely missing files.
- Each encoder/task is trained with seed 42; seed-specific artifacts live under `<run_slug>/seed_42/`, while `aggregate_summary.csv` and `aggregate_summary.json` keep the same summary schema for consistency.
- Validation and test exports include accuracy, macro F1, macro AUROC, macro AUPRC, per-class AUROC/AUPRC, classification reports, and confusion matrices where the metric is computable for the observed labels.
- CheXbert loading depends on the Stanford AIMI `chexbert.pth` checkpoint, and RadBERT depends on Hugging Face model availability. If either cannot be loaded in the current environment, the run is recorded in `skipped_runs.csv`.
- The best checkpoint saved for each encoder-task-seed run is selected by validation accuracy and written as `best_checkpoint.pt` inside that seed's artifact directory.
"""


# Assemble the ordered notebook cells for Experiment 02.
def build_cells() -> list[dict]:
    return [
        markdown_cell(CELL_0_MARKDOWN),
        markdown_cell(CELL_1_MARKDOWN),
        code_cell(CELL_2_CODE),
        markdown_cell(CELL_3_MARKDOWN),
        code_cell(CELL_4_CODE),
        markdown_cell(CELL_5_MARKDOWN),
        code_cell(CELL_6_CODE),
        markdown_cell(CELL_7_MARKDOWN),
        code_cell(CELL_8_CODE),
        markdown_cell(CELL_9_MARKDOWN),
        code_cell(CELL_10_CODE),
        markdown_cell(CELL_11_MARKDOWN),
        code_cell(CELL_12_CODE),
        markdown_cell(CELL_13_MARKDOWN),
        code_cell(CELL_14_CODE),
        markdown_cell(CELL_15_MARKDOWN),
    ]


# Write the generated benchmark notebook to the Experiments folder.
def main() -> None:
    source_notebook = resolve_source_notebook()
    source_nb = json.loads(source_notebook.read_text(encoding="utf-8"))
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
    print(f"Used source notebook: {source_notebook}")
    print(f"Wrote {TARGET_NOTEBOOK}")


if __name__ == "__main__":
    main()
