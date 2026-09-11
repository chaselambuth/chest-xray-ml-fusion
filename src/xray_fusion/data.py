# Data cleaning, table-building, vocabulary, and dataset helpers.

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


# Normalize a nullable text field into lowercase stripped text.
def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()

# Split NIH-style semicolon-delimited metadata into clean lowercase terms.
def split_semicolon_terms(text) -> list[str]:
    if pd.isna(text) or not str(text).strip():
        return []
    return [term.strip().lower() for term in str(text).split(";") if term.strip()]

# Map report metadata and impression text into the project label set.
def assign_clean_label(row) -> str:
    mesh_terms = split_semicolon_terms(row["MeSH"])
    problem_terms = split_semicolon_terms(row["Problems"])
    impression = str(row["impression"]).lower().strip()
    all_terms = set(mesh_terms + problem_terms)

    normal_phrases = [
        "no acute cardiopulmonary abnormality",
        "no acute cardiopulmonary abnormalities",
        "no acute cardiopulmonary findings",
        "no acute cardiopulmonary disease",
        "no acute cardiopulmonary process",
        "no acute pulmonary findings",
        "no acute pulmonary disease",
        "no acute pulmonary abnormality",
        "no evidence of active disease",
        "no active disease",
        "no acute findings",
        "no acute abnormality",
        "no acute process",
        "negative for acute abnormality",
        "clear lungs",
        "normal chest",
    ]
    if "normal" in all_terms or any(phrase in impression for phrase in normal_phrases):
        return "normal"
    if any("cardiomegaly" in term for term in all_terms) or "cardiomegaly" in impression or "cardiac shadow/enlarged" in all_terms:
        return "cardiomegaly"
    if any("pleural effusion" in term for term in all_terms) or "pleural effusion" in impression:
        return "pleural_effusion"

    chronic_terms = [
        "emphysema",
        "pulmonary emphysema",
        "pulmonary disease, chronic obstructive",
        "fibrosis",
        "interstitial fibrosis",
        "chronic obstructive",
        "copd",
        "bullous emphysema",
    ]
    if any(any(term in label for term in chronic_terms) for label in all_terms) or any(term in impression for term in chronic_terms):
        return "chronic_lung_disease"

    opacity_terms = ["opacity", "infiltrate", "airspace disease", "consolidation", "density"]
    if any(term in all_terms for term in opacity_terms) or any(term in impression for term in opacity_terms):
        return "pneumonia_or_opacity"
    return "other"


# Join frontal/lateral projections with reports and keep valid paired studies.
def build_study_level_table(
    projections: pd.DataFrame,
    reports: pd.DataFrame,
    image_dir: Path,
    text_columns: Iterable[str] = ("indication", "findings", "impression", "MeSH", "Problems", "comparison", "image"),
) -> pd.DataFrame:
    frontal_df = (
        projections.loc[projections["projection"].str.lower() == "frontal", ["uid", "filename"]]
        .rename(columns={"filename": "frontal_filename"})
        .copy()
    )
    lateral_df = (
        projections.loc[projections["projection"].str.lower() == "lateral", ["uid", "filename"]]
        .rename(columns={"filename": "lateral_filename"})
        .copy()
    )
    study_df = frontal_df.merge(lateral_df, on="uid", how="inner").merge(reports, on="uid", how="inner")
    study_df["frontal_path"] = study_df["frontal_filename"].apply(lambda name: str(image_dir / name))
    study_df["lateral_path"] = study_df["lateral_filename"].apply(lambda name: str(image_dir / name))

    for col in text_columns:
        if col in study_df.columns:
            study_df[col] = study_df[col].apply(clean_text)

    study_df["frontal_exists"] = study_df["frontal_path"].apply(lambda value: Path(value).exists())
    study_df["lateral_exists"] = study_df["lateral_path"].apply(lambda value: Path(value).exists())
    return study_df.loc[study_df["frontal_exists"] & study_df["lateral_exists"]].copy()


# Build a small whitespace-token vocabulary with pad and unknown entries.
def build_vocabulary(texts: Iterable[str], max_vocab: int = 5000) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(str(text).lower().split())
    vocab = {"<pad>": 0, "<unk>": 1}
    for idx, (word, _) in enumerate(counter.most_common(max_vocab - 2), start=2):
        vocab[word] = idx
    return vocab

# Encode text as fixed-length vocabulary ids for the simple fusion baseline.
def encode_text(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    tokens = str(text).lower().split()
    ids = [vocab.get(token, vocab["<unk>"]) for token in tokens[:max_len]]
    return ids + [vocab["<pad>"]] * max(0, max_len - len(ids))


# Dataset that returns frontal image, lateral image, and class label tuples.
class XRayDualImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        frontal = Image.open(row["frontal_path"]).convert("RGB")
        lateral = Image.open(row["lateral_path"]).convert("RGB")
        if self.transform is not None:
            frontal = self.transform(frontal)
            lateral = self.transform(lateral)
        label = torch.tensor(int(row["label_enc"]), dtype=torch.long)
        return frontal, lateral, label

# Dataset that tokenizes indication text for text-only transformer models.
class XRayTextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 64, label_key: str = "label"):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_key = label_key

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        text = clean_text(row.get("indication", ""))
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
            self.label_key: torch.tensor(int(row["label_enc"]), dtype=torch.long),
        }

# Dataset for fusion models, returning image tensors plus text features.
class XRayMultimodalDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer=None,
        image_transform=None,
        max_len: int = 64,
        encode_fn: Callable[[str], list[int]] | None = None,
        label_key: str = "label",
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_len = max_len
        self.encode_fn = encode_fn
        self.label_key = label_key

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        frontal = Image.open(row["frontal_path"]).convert("RGB")
        lateral = Image.open(row["lateral_path"]).convert("RGB")
        if self.image_transform is not None:
            frontal = self.image_transform(frontal)
            lateral = self.image_transform(lateral)

        label = torch.tensor(int(row["label_enc"]), dtype=torch.long)
        text = clean_text(row.get("indication", ""))
        if self.tokenizer is not None:
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
                self.label_key: label,
            }
        if self.encode_fn is not None:
            return frontal, lateral, torch.tensor(self.encode_fn(text), dtype=torch.long), label
        return frontal, lateral, label

# Dataset that returns dual-view image inputs in a Trainer-friendly dict.
class XRayDualImageDictDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_transform=None, label_key: str = "labels"):
        self.df = df.reset_index(drop=True)
        self.image_transform = image_transform
        self.label_key = label_key

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        frontal = Image.open(row["frontal_path"]).convert("RGB")
        lateral = Image.open(row["lateral_path"]).convert("RGB")
        if self.image_transform is not None:
            frontal = self.image_transform(frontal)
            lateral = self.image_transform(lateral)
        return {
            "frontal": frontal,
            "lateral": lateral,
            self.label_key: torch.tensor(int(row["label_enc"]), dtype=torch.long),
        }
