# Reusable inference helpers shared by the Flask app and demos.

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms
    from transformers import (
        AutoImageProcessor,
        AutoModelForSemanticSegmentation,
        AutoTokenizer,
    )
except Exception as exc:  # pragma: no cover - surfaced in the web UI
    torch = None
    nn = None
    F = None
    transforms = None
    AutoImageProcessor = None
    AutoModelForSemanticSegmentation = None
    AutoTokenizer = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
SPLIT_DIR = BASE_DIR / "splits"
HF_CACHE_DIR = BASE_DIR / "hf_cache"
EXPERIMENT_04_DIR = BASE_DIR / "Experiments" / "outputs" / "experiment_04_segmentation"

IMG_SIZE = 224
MAX_LEN = 64
TOKENIZER_NAME = "emilyalsentzer/Bio_ClinicalBERT"
PRETRAINED_SEGMENTER_MODEL = "Tianmu28/segformer-b0-segments-lungs-xray"

PIPELINE_SPECS = {
    "a_original_full": {
        "name": "Standard View",
        "summary_name": "A_original_full",
        "mode": "original",
        "description": "Original frontal and original lateral images.",
        "weights_path": EXPERIMENT_04_DIR / "a_original_full" / "best_checkpoint.pt",
    },
    "b_lung_cropped_frontal": {
        "name": "Lung-Focused View",
        "summary_name": "B_lung_cropped_frontal",
        "mode": "cropped_frontal",
        "description": "Focuses the frontal image around the lung field while keeping the lateral image unchanged.",
        "weights_path": EXPERIMENT_04_DIR / "b_lung_cropped_frontal" / "best_checkpoint.pt",
    },
    "c_lung_masked_dual": {
        "name": "Masked Lung View",
        "summary_name": "C_lung_masked_dual",
        "mode": "masked_dual",
        "description": "Applies a lung-field mask to both uploaded views before analysis.",
        "weights_path": EXPERIMENT_04_DIR / "c_lung_masked_dual" / "best_checkpoint.pt",
    },
}

DISPLAY_LABELS = {
    "cardiomegaly": "Cardiomegaly",
    "chronic_lung_disease": "Chronic Lung Disease",
    "normal": "Normal",
    "pleural_effusion": "Pleural Effusion",
    "pneumonia_or_opacity": "Pneumonia or Opacity",
}


# Raise a clear runtime error if the optional ML dependencies are unavailable.
def require_ml_stack() -> None:
    if IMPORT_ERROR is not None:
        raise RuntimeError(
            "The imaging models require torch, torchvision, transformers, "
            f"and matplotlib. Import failed with: {IMPORT_ERROR}"
        )

# Load saved torch weights across torch versions with and without weights_only.
def torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


if IMPORT_ERROR is None:
    # Capture image-branch activations and gradients for Grad-CAM overlays.
    class GradCAM:
        def __init__(self, model, target_layer):
            self.model = model
            self.activations = None
            self.gradients = None
            self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
            self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

        # Store forward activations from the hooked convolution.
        def _forward_hook(self, module, inputs, output):
            self.activations = output.detach()

        # Store backward gradients from the hooked convolution.
        def _backward_hook(self, module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Generate a normalized Grad-CAM map for the predicted or target class.
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

        # Remove hooks after a Grad-CAM request is complete.
        def close(self):
            self.forward_handle.remove()
            self.backward_handle.remove()

@lru_cache(maxsize=1)
# Read the experiment label order from the training split.
def load_label_names() -> Tuple[str, ...]:
    train_df = pd.read_csv(SPLIT_DIR / "train.csv")
    return tuple(sorted(train_df["clean_label"].dropna().unique().tolist()))

@lru_cache(maxsize=1)
# Load pipeline metrics when available, otherwise return configured defaults.
def load_pipeline_summary() -> pd.DataFrame:
    summary_path = EXPERIMENT_04_DIR / "pipeline_summary.csv"
    if summary_path.exists():
        return pd.read_csv(summary_path)
    return pd.DataFrame(
        [
            {
                "pipeline_slug": slug,
                "pipeline_name": spec["summary_name"],
                "description": spec["description"],
            "weights_path": str(spec["weights_path"]),
            }
            for slug, spec in PIPELINE_SPECS.items()
        ]
    )

@lru_cache(maxsize=3)
# Load and cache the tokenizer, saved model weights, transforms, and device for a pipeline.
def load_fusion_bundle(pipeline_slug: str):
    require_ml_stack()
    if pipeline_slug not in PIPELINE_SPECS:
        raise ValueError(f"Unknown pipeline: {pipeline_slug}")

    from xray_fusion.models import SelectedEncoderGatedFusionModel

    label_names = list(load_label_names())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = PIPELINE_SPECS[pipeline_slug]["weights_path"]
    saved_weights = torch_load(weights_path, map_location=device)
    encoder_spec = saved_weights.get(
        "encoder_spec",
        {
            "name": "Bio_ClinicalBERT",
            "slug": "bio_clinicalbert",
            "loader": "auto",
            "model_name": TOKENIZER_NAME,
            "tokenizer_name": TOKENIZER_NAME,
        },
    )
    tokenizer_name = encoder_spec.get("tokenizer_name") or encoder_spec.get("model_name") or TOKENIZER_NAME
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=HF_CACHE_DIR)

    model = SelectedEncoderGatedFusionModel(
        encoder_spec=encoder_spec,
        num_classes=len(label_names),
        freeze_bert=saved_weights.get("freeze_bert", True),
        freeze_image_backbone=saved_weights.get("freeze_image_backbone", False),
        pretrained_image_backbone=False,
        cache_dir=HF_CACHE_DIR,
    ).to(device)
    model.load_state_dict(saved_weights["model_state_dict"])
    model.eval()

    image_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    return {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "image_transform": image_transform,
        "label_names": label_names,
    }

@lru_cache(maxsize=1)
# Load and cache the lung segmentation model and image processor.
def load_segmenter_bundle():
    require_ml_stack()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(
        PRETRAINED_SEGMENTER_MODEL,
        cache_dir=HF_CACHE_DIR,
        use_fast=False,
    )
    model = AutoModelForSemanticSegmentation.from_pretrained(
        PRETRAINED_SEGMENTER_MODEL,
        cache_dir=HF_CACHE_DIR,
    ).to(device)
    model.eval()
    return {"processor": processor, "model": model, "device": device}


# Convert a stored label slug into user-facing text.
def pretty_label(label: str) -> str:
    return DISPLAY_LABELS.get(label, label.replace("_", " ").title())

# Return the tight bounding box around nonzero mask pixels, if any.
def largest_bbox_from_mask(mask_array: np.ndarray):
    ys, xs = np.where(mask_array > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

# Run the segmentation model and return a uint8 binary lung mask.
def infer_binary_lung_mask(image: Image.Image, threshold: float = 0.5) -> np.ndarray:
    bundle = load_segmenter_bundle()
    original_size = image.size
    model_inputs = bundle["processor"](images=image.convert("RGB"), return_tensors="pt")
    model_inputs = {key: value.to(bundle["device"]) for key, value in model_inputs.items()}

    with torch.no_grad():
        outputs = bundle["model"](**model_inputs)
        logits = outputs.logits

    upsampled = F.interpolate(
        logits,
        size=(original_size[1], original_size[0]),
        mode="bilinear",
        align_corners=False,
    )

    if upsampled.shape[1] == 1:
        probs = torch.sigmoid(upsampled)[0, 0].cpu().numpy()
        return (probs > threshold).astype(np.uint8) * 255

    probs = torch.softmax(upsampled, dim=1)[0].cpu().numpy()
    foreground = probs[1] if probs.shape[0] == 2 else probs[1:].sum(axis=0)
    return (foreground > threshold).astype(np.uint8) * 255


# Apply a binary mask to an RGB PIL image.
def masked_image(image: Image.Image, mask_array: np.ndarray) -> Image.Image:
    image_arr = np.array(image.convert("RGB"), dtype=np.uint8)
    mask01 = (mask_array > 0).astype(np.uint8)
    return Image.fromarray(image_arr * mask01[..., None])


# Crop an image to the lung mask box, falling back to the full image.
def cropped_frontal_image(image: Image.Image, mask_array: np.ndarray) -> Image.Image:
    bbox = largest_bbox_from_mask(mask_array)
    if bbox is None:
        return image.convert("RGB")
    return image.convert("RGB").crop(bbox)


# Blend a teal mask overlay onto an image for review.
def overlay_mask(image: Image.Image, mask_array: np.ndarray, alpha: float = 0.38) -> Image.Image:
    image_arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    mask = (mask_array > 0).astype(np.float32)
    color = np.zeros_like(image_arr)
    color[..., 0] = 0.08
    color[..., 1] = 0.62
    color[..., 2] = 0.67
    overlay = image_arr * (1 - alpha * mask[..., None]) + color * (alpha * mask[..., None])
    return Image.fromarray((np.clip(overlay, 0.0, 1.0) * 255).astype(np.uint8))


# Blend a Grad-CAM heatmap onto a resized image.
def overlay_gradcam(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = image.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
    base_arr = np.array(base, dtype=np.float32) / 255.0
    heat = plt.get_cmap("jet")(cam)[..., :3]
    overlay = (1 - alpha) * base_arr + alpha * heat
    return Image.fromarray((np.clip(overlay, 0.0, 1.0) * 255).astype(np.uint8))


# Convert a normalized Grad-CAM array into a heatmap image.
def cam_to_image(cam: np.ndarray) -> Image.Image:
    heat = plt.get_cmap("jet")(cam)[..., :3]
    return Image.fromarray((np.clip(heat, 0.0, 1.0) * 255).astype(np.uint8))


# Apply the selected experiment preprocessing pipeline to both views.
def apply_pipeline_preprocessing(
    frontal: Image.Image,
    lateral: Image.Image,
    pipeline_slug: str,
    threshold: float,
) -> Tuple[Image.Image, Image.Image, Dict[str, Optional[np.ndarray]]]:
    mode = PIPELINE_SPECS[pipeline_slug]["mode"]
    masks: Dict[str, Optional[np.ndarray]] = {"frontal": None, "lateral": None}

    if mode == "original":
        return frontal.convert("RGB"), lateral.convert("RGB"), masks

    frontal_mask = infer_binary_lung_mask(frontal, threshold=threshold)
    masks["frontal"] = frontal_mask

    if mode == "cropped_frontal":
        return cropped_frontal_image(frontal, frontal_mask), lateral.convert("RGB"), masks

    if mode == "masked_dual":
        lateral_mask = infer_binary_lung_mask(lateral, threshold=threshold)
        masks["lateral"] = lateral_mask
        return masked_image(frontal, frontal_mask), masked_image(lateral, lateral_mask), masks

    raise ValueError(f"Unknown pipeline mode: {mode}")

# Transform images and indication text into model-ready tensors.
def encode_inputs(bundle, frontal: Image.Image, lateral: Image.Image, indication: str):
    frontal_tensor = bundle["image_transform"](frontal).unsqueeze(0).to(bundle["device"])
    lateral_tensor = bundle["image_transform"](lateral).unsqueeze(0).to(bundle["device"])
    encoded = bundle["tokenizer"](
        indication or "",
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(bundle["device"])
    attention_mask = encoded["attention_mask"].to(bundle["device"])
    return frontal_tensor, lateral_tensor, input_ids, attention_mask


# Run preprocessing and model inference for one uploaded case.
def predict_case(
    frontal: Image.Image,
    lateral: Image.Image,
    indication: str,
    pipeline_slug: str,
    threshold: float,
):
    bundle = load_fusion_bundle(pipeline_slug)
    model_frontal, model_lateral, masks = apply_pipeline_preprocessing(
        frontal,
        lateral,
        pipeline_slug,
        threshold,
    )
    tensors = encode_inputs(bundle, model_frontal, model_lateral, indication)

    with torch.no_grad():
        logits = bundle["model"](*tensors)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    return {
        "probs": probs,
        "label_names": bundle["label_names"],
        "model_frontal": model_frontal,
        "model_lateral": model_lateral,
        "masks": masks,
    }

# Run inference and return Grad-CAM artifacts for one uploaded case.
def generate_gradcam(
    frontal: Image.Image,
    lateral: Image.Image,
    indication: str,
    pipeline_slug: str,
    threshold: float,
):
    bundle = load_fusion_bundle(pipeline_slug)
    model_frontal, model_lateral, masks = apply_pipeline_preprocessing(
        frontal,
        lateral,
        pipeline_slug,
        threshold,
    )
    tensors = encode_inputs(bundle, model_frontal, model_lateral, indication)
    grad_cam = GradCAM(bundle["model"], bundle["model"].image_encoder.backbone.layer4[-1].conv2)

    try:
        cam, logits = grad_cam.generate(*tensors)
    finally:
        grad_cam.close()

    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return {
        "cam": cam,
        "cam_image": cam_to_image(cam),
        "overlay": overlay_gradcam(model_frontal, cam),
        "probs": probs,
        "label_names": bundle["label_names"],
        "model_frontal": model_frontal,
        "model_lateral": model_lateral,
        "masks": masks,
    }

# Return masks and overlays for both uploaded X-ray views.
def run_segmentation(frontal: Image.Image, lateral: Image.Image, threshold: float):
    frontal_mask = infer_binary_lung_mask(frontal, threshold=threshold)
    lateral_mask = infer_binary_lung_mask(lateral, threshold=threshold)
    return {
        "frontal_mask": Image.fromarray(frontal_mask),
        "lateral_mask": Image.fromarray(lateral_mask),
        "frontal_overlay": overlay_mask(frontal, frontal_mask),
        "lateral_overlay": overlay_mask(lateral, lateral_mask),
    }


# Format the highest-probability class predictions for display.
def top_k_predictions(label_names: List[str], probs: np.ndarray, k: int) -> List[Dict[str, float | str]]:
    idxs = np.argsort(-probs)[:k]
    return [
        {
            "label": label_names[i],
            "display": pretty_label(label_names[i]),
            "score": float(probs[i]),
            "percent": f"{float(probs[i]):.1%}",
            "width": max(1.0, min(100.0, float(probs[i]) * 100.0)),
        }
        for i in idxs
    ]

# Map a probability into a plain-language confidence bucket.
def confidence_bucket(score: float) -> str:
    if score >= 0.90:
        return "very high"
    if score >= 0.75:
        return "high"
    if score >= 0.60:
        return "moderate"
    return "low"

# Create a short report-style summary for UI display.
def make_report(predicted_class: str, confidence: float, pipeline_name: str, indication: str) -> str:
    context_line = (
        "The entered indication was included as supporting clinical context."
        if indication.strip()
        else "No indication text was supplied."
    )
    return (
        f"The imaging assistant predicts {pretty_label(predicted_class)} with "
        f"{confidence_bucket(confidence)} confidence ({confidence:.1%}) using the "
        f"{pipeline_name.lower()} processing profile. {context_line} This output is "
        "intended to support review and is not a standalone diagnosis."
    )
