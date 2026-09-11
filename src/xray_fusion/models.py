# Model backbones, classifiers, and fusion blocks used by the experiments.

from __future__ import annotations

from pathlib import Path
import shutil

import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoModel, BertConfig, BertModel
from transformers.utils.hub import cached_file


# Load a torch checkpoint across torch versions with and without weights_only.
def torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

# Load CheXbert checkpoint weights into a BertModel backbone.
def load_chexbert_backbone(repo_id: str, filename: str, cache_dir: Path | None = None):
    checkpoint_path = cached_file(repo_id, filename, cache_dir=cache_dir)
    checkpoint = torch_load(Path(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    bert_state = {
        key[len("bert."):]: value
        for key, value in state_dict.items()
        if key.startswith("bert.")
    }
    bert = BertModel(BertConfig())
    bert.load_state_dict(bert_state, strict=False)
    return bert

# Load a text encoder from either the CheXbert checkpoint or AutoModel.
def load_text_backbone(encoder_spec: dict, cache_dir: Path | None = None):
    if encoder_spec.get("loader") == "chexbert":
        return load_chexbert_backbone(
            encoder_spec["repo_id"],
            encoder_spec["filename"],
            cache_dir=cache_dir,
        )
    return AutoModel.from_pretrained(encoder_spec["model_name"], cache_dir=cache_dir)

# Extract a CLS-style embedding from common transformer output objects.
def get_cls_embedding(outputs):
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state[:, 0, :]
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    raise ValueError("Unsupported transformer output: unable to extract CLS embedding.")

# Return the Hugging Face cache directory name for a model repository.
def hf_repo_cache_dir(repo_id: str, cache_dir: Path) -> Path:
    return Path(cache_dir) / f"models--{repo_id.replace('/', '--')}"

# List local Hugging Face snapshot directories for a repository.
def iter_hf_snapshots(repo_id: str, cache_dir: Path):
    snapshots_dir = hf_repo_cache_dir(repo_id, cache_dir) / "snapshots"
    if not snapshots_dir.exists():
        return []
    return sorted(path for path in snapshots_dir.iterdir() if path.is_dir())

# Find the first nonempty cached file matching one of the requested names.
def first_cached_file(repo_id: str, filenames, cache_dir: Path):
    for filename in filenames:
        for snapshot_dir in iter_hf_snapshots(repo_id, cache_dir):
            candidate = snapshot_dir / filename
            if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
    return None

# Copy a cached file unless the target already has the same byte size.
def copy_if_needed(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and target_path.stat().st_size == source_path.stat().st_size:
        return
    shutil.copy2(source_path, target_path)

# Copy enough cached HF files into a stable local directory for offline loading.
def materialize_cached_repo(repo_id: str, required_files, optional_files=None, cache_dir: Path | None = None, resolved_dir: Path | None = None):
    optional_files = optional_files or []
    if cache_dir is None or resolved_dir is None:
        return None
    target_dir = Path(resolved_dir) / repo_id.replace("/", "__")
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
    return None if missing else target_dir

# Resolve a cached model source path when required model files are present.
def resolve_model_source(model_name: str, cache_dir: Path, resolved_dir: Path):
    return materialize_cached_repo(
        model_name,
        required_files=["config.json", ["model.safetensors", "pytorch_model.bin"]],
        optional_files=["vocab.txt", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
        cache_dir=cache_dir,
        resolved_dir=resolved_dir,
    )

# Resolve a cached tokenizer source path when required tokenizer files exist.
def resolve_tokenizer_source(tokenizer_name: str, cache_dir: Path, resolved_dir: Path):
    return materialize_cached_repo(
        tokenizer_name,
        required_files=["vocab.txt"],
        optional_files=["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
        cache_dir=cache_dir,
        resolved_dir=resolved_dir,
    )

# Load a Hugging Face AutoModel, preferring local resolved files when available.
def load_auto_backbone(model_name: str, cache_dir: Path | None = None, resolved_dir: Path | None = None):
    local_source = resolve_model_source(model_name, cache_dir, resolved_dir) if cache_dir and resolved_dir else None
    model_source = local_source if local_source is not None else model_name
    local_only = local_source is not None
    try:
        return AutoModel.from_pretrained(model_source, cache_dir=cache_dir, local_files_only=local_only, use_safetensors=True)
    except Exception:
        return AutoModel.from_pretrained(model_source, cache_dir=cache_dir, local_files_only=local_only, use_safetensors=False)


# Encode frontal and lateral X-rays with a shared ResNet18 backbone.
class DualImageEncoder(nn.Module):
    def __init__(self, freeze_backbone: bool = False, pretrained_backbone: bool = True):
        super().__init__()
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained_backbone else None
            backbone = models.resnet18(weights=weights)
        except AttributeError:
            backbone = models.resnet18(pretrained=pretrained_backbone)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.output_dim = feature_dim * 2
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, frontal, lateral):
        frontal_features = self.backbone(frontal)
        lateral_features = self.backbone(lateral)
        return torch.cat([frontal_features, lateral_features], dim=1)

# Image-only classifier over concatenated frontal and lateral features.
class DualImageClassifier(nn.Module):
    def __init__(self, num_classes: int, freeze_backbone: bool = False, dropout: float = 0.2):
        super().__init__()
        self.image_encoder = DualImageEncoder(freeze_backbone=freeze_backbone)
        self.classifier = nn.Sequential(
            nn.Linear(self.image_encoder.output_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, frontal, lateral):
        return self.classifier(self.image_encoder(frontal, lateral))


# Baseline fusion model using dual-image features and averaged token embeddings.
class SimpleVocabularyFusionClassifier(nn.Module):
    def __init__(self, num_classes: int, vocab_size: int, text_embed_dim: int = 64, text_hidden_dim: int = 64):
        super().__init__()
        self.image_encoder = DualImageEncoder()
        self.text_embedding = nn.Embedding(vocab_size, text_embed_dim, padding_idx=0)
        self.text_proj = nn.Sequential(
            nn.Linear(text_embed_dim, text_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.image_encoder.output_dim + text_hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, frontal, lateral, text_ids):
        image_features = self.image_encoder(frontal, lateral)
        text_features = self.text_proj(self.text_embedding(text_ids).mean(dim=1))
        return self.classifier(torch.cat([image_features, text_features], dim=1))

# Text-only classifier built on the selected clinical transformer encoder.
class TextOnlyClassifier(nn.Module):
    def __init__(self, encoder_spec: dict, num_classes: int, freeze_bert: bool = False, cache_dir: Path | None = None, resolved_dir: Path | None = None):
        super().__init__()
        self.encoder_spec = encoder_spec
        if encoder_spec.get("loader") == "auto":
            self.bert = load_auto_backbone(encoder_spec["model_name"], cache_dir=cache_dir, resolved_dir=resolved_dir)
        else:
            self.bert = load_chexbert_backbone(encoder_spec["repo_id"], encoder_spec["filename"], cache_dir=cache_dir)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.bert.config.hidden_size, num_classes))

    # Return the raw transformer CLS embedding for input text.
    def encode_text(self, input_ids, attention_mask):
        return get_cls_embedding(self.bert(input_ids=input_ids, attention_mask=attention_mask))

    def forward(self, input_ids, attention_mask):
        return self.classifier(self.encode_text(input_ids, attention_mask))

# Concatenation-based dual-image plus transformer text fusion classifier.
class FusionClassifier(nn.Module):
    def __init__(
        self,
        encoder_spec: dict,
        num_classes: int,
        freeze_bert: bool = False,
        freeze_image_backbone: bool = False,
        cache_dir: Path | None = None,
        resolved_dir: Path | None = None,
    ):
        super().__init__()
        self.encoder_spec = encoder_spec
        self.image_encoder = DualImageEncoder(freeze_backbone=freeze_image_backbone)
        if encoder_spec.get("loader") == "auto":
            self.bert = load_auto_backbone(encoder_spec["model_name"], cache_dir=cache_dir, resolved_dir=resolved_dir)
        else:
            self.bert = load_chexbert_backbone(encoder_spec["repo_id"], encoder_spec["filename"], cache_dir=cache_dir)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.text_proj = nn.Sequential(nn.Linear(self.bert.config.hidden_size, 256), nn.ReLU(), nn.Dropout(0.2))
        self.classifier = nn.Sequential(
            nn.Linear(self.image_encoder.output_dim + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    # Project the transformer CLS embedding into the fusion feature space.
    def encode_text(self, input_ids, attention_mask):
        return self.text_proj(get_cls_embedding(self.bert(input_ids=input_ids, attention_mask=attention_mask)))

    def forward(self, frontal, lateral, input_ids, attention_mask):
        image_features = self.image_encoder(frontal, lateral)
        text_features = self.encode_text(input_ids, attention_mask)
        return self.classifier(torch.cat([image_features, text_features], dim=1))


# Fusion classifier that accepts a preconstructed transformer backbone.
class TransformerFusionClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        text_backbone,
        freeze_bert: bool = True,
        freeze_image_backbone: bool = False,
        text_dropout: float = 0.2,
        classifier_dropout: float = 0.3,
    ):
        super().__init__()
        self.image_encoder = DualImageEncoder(freeze_backbone=freeze_image_backbone)
        self.bert = text_backbone
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        hidden_size = self.bert.config.hidden_size
        self.text_proj = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(text_dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.image_encoder.output_dim + 256, 256),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, frontal, lateral, input_ids, attention_mask):
        image_features = self.image_encoder(frontal, lateral)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = self.text_proj(outputs.last_hidden_state[:, 0, :])
        return self.classifier(torch.cat([image_features, text_features], dim=1))

# Fusion classifier specialized to a Bio_ClinicalBERT model path or name.
class BioClinicalBertFusionClassifier(TransformerFusionClassifier):
    def __init__(
        self,
        num_classes: int,
        model_name_or_path: str,
        cache_dir: Path | None = None,
        freeze_bert: bool = True,
        freeze_image_backbone: bool = False,
        text_dropout: float = 0.2,
        classifier_dropout: float = 0.3,
    ):
        model_path = Path(str(model_name_or_path))
        text_backbone = AutoModel.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            use_safetensors=True,
            local_files_only=model_path.exists(),
        )
        super().__init__(
            num_classes=num_classes,
            text_backbone=text_backbone,
            freeze_bert=freeze_bert,
            freeze_image_backbone=freeze_image_backbone,
            text_dropout=text_dropout,
            classifier_dropout=classifier_dropout,
        )


# Fuse image and text features with learned residual gates and interactions.
class ResidualGatedFusionBlock(nn.Module):
    def __init__(self, image_dim: int, text_dim: int, fusion_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.image_proj = nn.Sequential(nn.Linear(image_dim, fusion_dim), nn.ReLU(), nn.Dropout(dropout))
        self.text_proj = nn.Sequential(nn.Linear(text_dim, fusion_dim), nn.ReLU(), nn.Dropout(dropout))
        self.gate = nn.Sequential(nn.Linear(fusion_dim * 2, fusion_dim), nn.ReLU(), nn.Linear(fusion_dim, fusion_dim), nn.Sigmoid())
        self.output_dim = fusion_dim * 3

    # Return fused features plus auxiliary hidden states and gate weights.
    def forward(self, image_features, text_features):
        image_hidden = self.image_proj(image_features)
        text_hidden = self.text_proj(text_features)
        gate = self.gate(torch.cat([image_hidden, text_hidden], dim=1))
        gated_image = gate * image_hidden
        gated_text = (1.0 - gate) * text_hidden
        interaction = gated_image * gated_text
        fused = torch.cat([gated_image, gated_text, interaction], dim=1)
        gate_weights = torch.stack([gate.mean(dim=1), (1.0 - gate).mean(dim=1)], dim=1)
        return {
            "fused_features": fused,
            "image_hidden": image_hidden,
            "text_hidden": text_hidden,
            "gate_weights": gate_weights,
        }

# End-to-end gated fusion model for the selected text encoder specification.
class SelectedEncoderGatedFusionModel(nn.Module):
    def __init__(
        self,
        encoder_spec: dict,
        num_classes: int,
        fusion_block=None,
        fusion_dim: int = 256,
        fusion_dropout: float = 0.2,
        freeze_bert: bool = False,
        freeze_image_backbone: bool = False,
        pretrained_image_backbone: bool = True,
        cache_dir: Path | None = None,
        resolved_dir: Path | None = None,
    ):
        super().__init__()
        self.encoder_spec = encoder_spec
        self.image_encoder = DualImageEncoder(
            freeze_backbone=freeze_image_backbone,
            pretrained_backbone=pretrained_image_backbone,
        )
        if encoder_spec.get("loader") == "auto":
            self.bert = load_auto_backbone(encoder_spec["model_name"], cache_dir=cache_dir, resolved_dir=resolved_dir)
        else:
            self.bert = load_chexbert_backbone(encoder_spec["repo_id"], encoder_spec["filename"], cache_dir=cache_dir)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.fusion_block = fusion_block or ResidualGatedFusionBlock(
            image_dim=self.image_encoder.output_dim,
            text_dim=self.bert.config.hidden_size,
            fusion_dim=fusion_dim,
            dropout=fusion_dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_block.output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    # Return the selected encoder's CLS-style text representation.
    def encode_text(self, input_ids, attention_mask):
        return get_cls_embedding(self.bert(input_ids=input_ids, attention_mask=attention_mask))

    def forward(self, frontal, lateral, input_ids, attention_mask, return_aux: bool = False):
        image_features = self.image_encoder(frontal, lateral)
        text_features = self.encode_text(input_ids, attention_mask)
        aux = self.fusion_block(image_features, text_features)
        logits = self.classifier(aux["fused_features"])
        if return_aux:
            return logits, aux
        return logits
