# Streamlit workbench for XRAY Fusion prediction, segmentation, and Grad-CAM.

from html import escape
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from transformers import (
        AutoImageProcessor,
        AutoModelForSemanticSegmentation,
        AutoTokenizer,
    )
except Exception as exc:
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

# Page config
st.set_page_config(
    page_title="XRAY Fusion Workbench",
    page_icon="XR",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Local model artifact configuration.
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
SPLIT_DIR = BASE_DIR / "splits"
HF_CACHE_DIR = BASE_DIR / "hf_cache"
EXPERIMENT_04_DIR = BASE_DIR / "Experiments" / "outputs" / "experiment_04_segmentation"

IMG_SIZE = 224
SEG_IMG_SIZE = 512
MAX_LEN = 64
TOKENIZER_NAME = "emilyalsentzer/Bio_ClinicalBERT"
PRETRAINED_SEGMENTER_MODEL = "Tianmu28/segformer-b0-segments-lungs-xray"

PIPELINE_SPECS = {
    "a_original_full": {
        "name": "Recommended Full-Image Model",
        "summary_name": "A_original_full",
        "mode": "original",
        "description": "Uses the original frontal and lateral images. This profile had the best balanced test result and is the recommended default.",
        "weights_path": EXPERIMENT_04_DIR / "a_original_full" / "best_checkpoint.pt",
    },
    "b_lung_cropped_frontal": {
        "name": "Lung-Cropped Frontal Model",
        "summary_name": "B_lung_cropped_frontal",
        "mode": "cropped_frontal",
        "description": "Crops the frontal image around the lung field and keeps the lateral image unchanged. This profile had higher accuracy but lower balanced performance.",
        "weights_path": EXPERIMENT_04_DIR / "b_lung_cropped_frontal" / "best_checkpoint.pt",
    },
    "c_lung_masked_dual": {
        "name": "Masked Lung Model",
        "summary_name": "C_lung_masked_dual",
        "mode": "masked_dual",
        "description": "Applies a lung-field mask to both uploaded views. This profile performed poorly and is kept for comparison.",
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


# Inject the custom CSS used by the Streamlit interface.
def inject_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg: #f6f8fb;
                --panel: #ffffff;
                --panel-soft: #f9fbfc;
                --ink: #18232f;
                --muted: #657485;
                --line: #dbe4ea;
                --teal: #177e89;
                --teal-dark: #0f5e67;
                --blue: #315f8f;
                --amber: #b7791f;
            }

            .stApp {
                background: var(--bg);
                color: var(--ink);
            }

            .block-container {
                max-width: 1260px;
                padding-top: 1.25rem;
                padding-bottom: 2rem;
            }

            [data-testid="stSidebar"] {
                background: #edf3f6;
                border-right: 1px solid var(--line);
            }

            .app-header {
                border: 1px solid var(--line);
                border-radius: 8px;
                background:
                    linear-gradient(135deg, #ffffff 0%, #f7fbfc 58%, #fff7e8 100%);
                padding: 1.25rem 1.35rem;
                margin-bottom: 1rem;
            }

            .eyebrow {
                color: var(--teal-dark);
                font-size: .78rem;
                font-weight: 700;
                letter-spacing: .06rem;
                text-transform: uppercase;
                margin-bottom: .3rem;
            }

            .app-title {
                color: var(--ink);
                font-size: clamp(1.8rem, 2.8vw, 2.7rem);
                font-weight: 760;
                line-height: 1.08;
                margin: 0;
            }

            .app-subtitle {
                max-width: 820px;
                color: var(--muted);
                font-size: 1rem;
                line-height: 1.55;
                margin: .7rem 0 0;
            }

            .status-row {
                display: flex;
                flex-wrap: wrap;
                gap: .45rem;
                margin-top: 1rem;
            }

            .status-pill {
                align-items: center;
                border: 1px solid var(--line);
                border-radius: 999px;
                background: rgba(255, 255, 255, .78);
                color: #344456;
                display: inline-flex;
                font-size: .82rem;
                font-weight: 650;
                min-height: 30px;
                padding: .28rem .68rem;
                white-space: nowrap;
            }

            .risk-banner {
                border: 1px solid #efd9a9;
                border-left: 4px solid var(--amber);
                border-radius: 8px;
                background: #fffaf0;
                color: #563b14;
                line-height: 1.45;
                padding: .85rem 1rem;
                margin: .35rem 0 1rem;
            }

            .metric-strip {
                display: grid;
                gap: .75rem;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                margin-bottom: 1rem;
            }

            .metric-tile {
                border: 1px solid var(--line);
                border-radius: 8px;
                background: #ffffff;
                padding: .85rem .9rem;
            }

            .metric-label {
                color: var(--muted);
                font-size: .78rem;
                font-weight: 700;
                text-transform: uppercase;
            }

            .metric-value {
                color: var(--ink);
                font-size: 1.22rem;
                font-weight: 760;
                line-height: 1.25;
                margin-top: .22rem;
                overflow-wrap: anywhere;
            }

            .confidence-list {
                display: grid;
                gap: .65rem;
                margin: .25rem 0 .4rem;
            }

            .confidence-row {
                display: grid;
                gap: .5rem;
            }

            .confidence-head {
                align-items: baseline;
                display: flex;
                justify-content: space-between;
                gap: 1rem;
                color: var(--ink);
                font-size: .92rem;
                font-weight: 680;
            }

            .confidence-score {
                color: var(--muted);
                font-variant-numeric: tabular-nums;
            }

            .confidence-track {
                background: #e7edf1;
                border-radius: 999px;
                height: .55rem;
                overflow: hidden;
            }

            .confidence-fill {
                background: linear-gradient(90deg, var(--teal), var(--blue));
                border-radius: inherit;
                height: 100%;
            }

            .report-box {
                border: 1px solid #cfdce4;
                border-radius: 8px;
                background: #f8fbfc;
                color: #2e3d4d;
                line-height: 1.55;
                padding: .95rem 1rem;
            }

            .empty-state {
                border: 1px dashed #b9c8d2;
                border-radius: 8px;
                background: #f9fcfd;
                color: var(--muted);
                display: flex;
                min-height: 260px;
                align-items: center;
                justify-content: center;
                text-align: center;
                padding: 1rem;
            }

            .small-note {
                color: var(--muted);
                font-size: .86rem;
                line-height: 1.42;
            }

            div.stButton > button {
                border-radius: 6px;
                min-height: 2.8rem;
                font-weight: 760;
                width: 100%;
            }

            div.stButton > button[kind="primary"] {
                background: var(--teal);
                border-color: var(--teal);
            }

            div.stButton > button[kind="primary"]:hover {
                background: var(--teal-dark);
                border-color: var(--teal-dark);
            }

            [data-testid="stFileUploader"] {
                border: 1px dashed #b8c7d1;
                border-radius: 8px;
                padding: .55rem;
                background: #fbfdfe;
            }

            @media (max-width: 900px) {
                .metric-strip {
                    grid-template-columns: 1fr;
                }

                .app-header {
                    padding: 1rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Capture image-branch activations and gradients for Grad-CAM overlays.
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
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


# Raise a clear runtime error if the optional ML stack failed to import.
def require_torch_stack() -> None:
    if TORCH_IMPORT_ERROR is not None:
        raise RuntimeError(
            "The imaging models require torch, torchvision, transformers, "
            f"and matplotlib. Import failed with: {TORCH_IMPORT_ERROR}"
        )


# Load saved torch weights across torch versions with and without weights_only.
def torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


@st.cache_data(show_spinner=False)
# Read the label order from the training split.
def load_label_names() -> List[str]:
    train_df = pd.read_csv(SPLIT_DIR / "train.csv")
    return sorted(train_df["clean_label"].dropna().unique().tolist())


@st.cache_data(show_spinner=False)
# Load pipeline metrics when available, otherwise return configured defaults.
def load_pipeline_summary() -> pd.DataFrame:
    summary_path = EXPERIMENT_04_DIR / "pipeline_summary.csv"
    if summary_path.exists():
        return pd.read_csv(summary_path)
    rows = []
    for slug, spec in PIPELINE_SPECS.items():
        rows.append(
            {
                "pipeline_slug": slug,
                "pipeline_name": spec["summary_name"],
                "description": spec["description"],
                "weights_path": str(spec["weights_path"]),
            }
        )
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner="Loading fusion model...")
# Load and cache the tokenizer, saved model weights, transforms, and device for a pipeline.
def load_fusion_bundle(pipeline_slug: str):
    require_torch_stack()
    if pipeline_slug not in PIPELINE_SPECS:
        raise ValueError(f"Unknown pipeline: {pipeline_slug}")

    from xray_fusion.models import SelectedEncoderGatedFusionModel

    label_names = load_label_names()
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


@st.cache_resource(show_spinner="Loading lung segmentation model...")
# Load and cache the lung segmentation model and image processor.
def load_segmenter_bundle():
    require_torch_stack()
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


# Convert an uploaded Streamlit file into an RGB PIL image, if present.
def load_upload(uploaded_file) -> Optional[Image.Image]:
    if uploaded_file is None:
        return None
    return Image.open(uploaded_file).convert("RGB")


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
        "bundle": bundle,
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
        "overlay": overlay_gradcam(model_frontal, cam),
        "probs": probs,
        "label_names": bundle["label_names"],
        "model_frontal": model_frontal,
        "model_lateral": model_lateral,
        "masks": masks,
    }


# Return the highest-probability class labels and scores.
def top_k_predictions(label_names: List[str], probs: np.ndarray, k: int) -> List[Tuple[str, float]]:
    idxs = np.argsort(-probs)[:k]
    return [(label_names[i], float(probs[i])) for i in idxs]


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
        " The entered indication was included through the Bio_ClinicalBERT text branch."
        if indication.strip()
        else " No indication text was supplied to the text branch."
    )
    return (
        f"The imaging model predicts {pretty_label(predicted_class)} with "
        f"{confidence_bucket(confidence)} confidence ({confidence:.1%}) using the "
        f"{pipeline_name} profile.{context_line} This is a research output and is not "
        "a clinical diagnosis."
    )


# Render the page header and status pills.
def render_header() -> None:
    st.markdown(
        """
        <section class="app-header">
            <div class="eyebrow">XRAY Fusion Project</div>
            <h1 class="app-title">Chest X-Ray Fusion Workbench</h1>
            <p class="app-subtitle">
                Dual-view chest X-ray inference using the best balanced model profile,
                optional lung segmentation preprocessing, and Grad-CAM interpretation.
            </p>
            <div class="status-row">
                <span class="status-pill">Recommended profile</span>
                <span class="status-pill">Bio_ClinicalBERT + ResNet18</span>
                <span class="status-pill">Full-image default</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


# Render the research-use warning shown above the controls.
def render_risk_banner() -> None:
    st.markdown(
        """
        <div class="risk-banner">
            This interface runs the selected research model locally. Results are model
            predictions for review and are not medical advice.
        </div>
        """,
        unsafe_allow_html=True,
    )


# Render compact probability bars for top class predictions.
def render_probability_bars(items: List[Tuple[str, float]]) -> None:
    for name, score in items:
        label_col, score_col = st.columns([3, 1])
        with label_col:
            st.markdown(f"**{pretty_label(name)}**")
        with score_col:
            st.markdown(f"{score:.1%}")
        st.progress(float(score))


# Render the top-class, confidence, and model-profile metric tiles.
def render_metric_strip(pred_name: str, pred_score: float, pipeline_label: str) -> None:
    st.markdown(
        f"""
        <div class="metric-strip">
            <div class="metric-tile">
                <div class="metric-label">Top class</div>
                <div class="metric-value">{escape(pretty_label(pred_name))}</div>
            </div>
            <div class="metric-tile">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{pred_score:.1%}</div>
            </div>
            <div class="metric-tile">
                <div class="metric-label">Model profile</div>
                <div class="metric-value">{escape(pipeline_label)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Render the available pipeline summary metrics in the sidebar.
def render_pipeline_table(summary_df: pd.DataFrame) -> None:
    display_df = summary_df.copy()
    if "pipeline_slug" in display_df.columns:
        display_df["pipeline_name"] = display_df["pipeline_slug"].map(
            lambda slug: PIPELINE_SPECS.get(slug, {}).get("name", slug)
        )
    if "test_macro_f1" in display_df.columns:
        display_df = display_df.sort_values("test_macro_f1", ascending=False)
    cols = [
        "pipeline_name",
        "best_epoch",
        "val_accuracy",
        "test_accuracy",
        "test_macro_f1",
    ]
    available_cols = [col for col in cols if col in display_df.columns]
    if not available_cols:
        return
    st.dataframe(
        display_df[available_cols],
        width="stretch",
        hide_index=True,
        column_config={
            "pipeline_name": "Model profile",
            "best_epoch": "Selected epoch",
            "val_accuracy": st.column_config.NumberColumn("Val acc", format="%.3f"),
            "test_accuracy": st.column_config.NumberColumn("Test acc", format="%.3f"),
            "test_macro_f1": st.column_config.NumberColumn("Test macro F1", format="%.3f"),
        },
    )


# Show an image preview or a consistent empty upload placeholder.
def render_image_or_empty(image: Optional[Image.Image], message: str, caption: str) -> None:
    if image is None:
        st.markdown(
            f"""
            <div class="empty-state">
                <div>{escape(message)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    st.image(image, caption=caption, width="stretch")


# Render probabilities, report text, and model inputs for prediction mode.
def render_prediction_output(result, top_k: int, pipeline_label: str, indication: str) -> None:
    top_preds = top_k_predictions(result["label_names"], result["probs"], top_k)
    pred_name, pred_score = top_preds[0]

    render_metric_strip(pred_name, pred_score, pipeline_label)
    left, right = st.columns([1, 1], gap="large")

    with left:
        with st.container(border=True):
            st.markdown("#### Class Probabilities")
            render_probability_bars(top_preds)
            report = make_report(pred_name, pred_score, pipeline_label, indication)
            st.markdown(
                f"""
                <div class="report-box">
                    <strong>Report-style output</strong><br>
                    {escape(report)}
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        with st.container(border=True):
            st.markdown("#### Model Inputs")
            cols = st.columns(2)
            with cols[0]:
                st.image(result["model_frontal"], caption="Frontal model input", width="stretch")
            with cols[1]:
                st.image(result["model_lateral"], caption="Lateral model input", width="stretch")

# Render lung masks and overlays for both uploaded views.
def render_segmentation_output(frontal: Image.Image, lateral: Image.Image, threshold: float) -> None:
    with st.spinner("Running segmentation model..."):
        frontal_mask = infer_binary_lung_mask(frontal, threshold=threshold)
        lateral_mask = infer_binary_lung_mask(lateral, threshold=threshold)

    st.markdown("#### Segmentation")
    row1 = st.columns(3)
    with row1[0]:
        st.image(frontal, caption="Frontal original", width="stretch")
    with row1[1]:
        st.image(Image.fromarray(frontal_mask), caption="Frontal mask", width="stretch")
    with row1[2]:
        st.image(overlay_mask(frontal, frontal_mask), caption="Frontal overlay", width="stretch")

    row2 = st.columns(3)
    with row2[0]:
        st.image(lateral, caption="Lateral original", width="stretch")
    with row2[1]:
        st.image(Image.fromarray(lateral_mask), caption="Lateral mask", width="stretch")
    with row2[2]:
        st.image(overlay_mask(lateral, lateral_mask), caption="Lateral overlay", width="stretch")


# Render Grad-CAM probabilities, heatmap, and overlay outputs.
def render_gradcam_output(result, top_k: int, pipeline_label: str) -> None:
    top_preds = top_k_predictions(result["label_names"], result["probs"], top_k)
    pred_name, pred_score = top_preds[0]

    render_metric_strip(pred_name, pred_score, pipeline_label)
    left, right = st.columns([0.95, 1.05], gap="large")

    with left:
        with st.container(border=True):
            st.markdown("#### Grad-CAM Probabilities")
            render_probability_bars(top_preds)
            st.markdown(
                '<p class="small-note">Grad-CAM is generated from the frontal branch target layer: '
                '<code>image_encoder.backbone.layer4[-1].conv2</code>.</p>',
                unsafe_allow_html=True,
            )

    with right:
        cols = st.columns(3)
        with cols[0]:
            st.image(result["model_frontal"].resize((IMG_SIZE, IMG_SIZE)), caption="Frontal model input", width="stretch")
        with cols[1]:
            st.image(result["cam"], caption="Grad-CAM", clamp=True, width="stretch")
        with cols[2]:
            st.image(result["overlay"], caption="Overlay", width="stretch")

# Render sidebar controls and return the selected run configuration.
def render_sidebar() -> Tuple[str, str, int, float]:
    summary_df = load_pipeline_summary()

    st.sidebar.title("Model Controls")
    mode = st.sidebar.radio(
        "View",
        ["Prediction", "Segmentation", "Grad-CAM"],
        horizontal=False,
    )
    pipeline_slug = st.sidebar.selectbox(
        "Model profile",
        options=list(PIPELINE_SPECS.keys()),
        format_func=lambda slug: PIPELINE_SPECS[slug]["name"],
        index=0,
    )
    top_k = st.sidebar.slider("Prediction count", min_value=1, max_value=5, value=5)
    threshold = st.sidebar.slider("Segmentation threshold", 0.10, 0.90, 0.50, 0.05)

    st.sidebar.divider()
    st.sidebar.subheader("Run Summary")
    render_pipeline_table(summary_df)

    return mode, pipeline_slug, top_k, threshold

# Render upload and indication controls plus the image preview panel.
def render_inputs() -> Tuple[Optional[Image.Image], Optional[Image.Image], str, bool]:
    st.markdown("### Case Inputs")
    input_col, preview_col = st.columns([0.95, 1.05], gap="large")
    input_version = st.session_state.setdefault("case_input_version", 0)

    with input_col:
        with st.container(border=True):
            frontal_file = st.file_uploader(
                "Upload frontal chest X-ray",
                type=["png", "jpg", "jpeg"],
                key=f"frontal_upload_{input_version}",
            )
            lateral_file = st.file_uploader(
                "Upload lateral chest X-ray",
                type=["png", "jpg", "jpeg"],
                key=f"lateral_upload_{input_version}",
            )
            indication = st.text_area(
                "Indication text",
                placeholder="Example: chest pain, shortness of breath, cough",
                height=110,
                key=f"indication_text_{input_version}",
            )
            button_cols = st.columns([1, 1])
            with button_cols[0]:
                run_case = st.button("Run Selected View", type="primary")
            with button_cols[1]:
                start_new_case = st.button("Start New Case")

            if start_new_case:
                st.session_state["case_input_version"] = input_version + 1
                st.rerun()

    frontal = load_upload(frontal_file)
    lateral = load_upload(lateral_file)

    with preview_col:
        with st.container(border=True):
            st.markdown("#### Image Review")
            cols = st.columns(2)
            with cols[0]:
                render_image_or_empty(frontal, "Upload a frontal image.", "Frontal image")
            with cols[1]:
                render_image_or_empty(lateral, "Upload a lateral image.", "Lateral image")

    return frontal, lateral, indication, run_case

# Render the full Streamlit app and route the selected analysis mode.
def render_app() -> None:
    inject_styles()
    render_header()
    render_risk_banner()

    if TORCH_IMPORT_ERROR is not None:
        st.error(
            "The app cannot load the imaging models because a required ML package failed "
            f"to import: {TORCH_IMPORT_ERROR}"
        )
        return

    mode, pipeline_slug, top_k, threshold = render_sidebar()
    pipeline_label = PIPELINE_SPECS[pipeline_slug]["name"]
    pipeline_description = PIPELINE_SPECS[pipeline_slug]["description"]

    st.markdown(
        f"""
        <p class="small-note">
            Active model profile: <strong>{escape(pipeline_label)}</strong>. {escape(pipeline_description)}
        </p>
        """,
        unsafe_allow_html=True,
    )

    frontal, lateral, indication, run_case = render_inputs()

    if not run_case:
        return

    if frontal is None or lateral is None:
        st.error("Please upload both frontal and lateral chest X-ray images.")
        return

    try:
        if mode == "Segmentation":
            render_segmentation_output(frontal, lateral, threshold)
        elif mode == "Grad-CAM":
            with st.spinner("Running model and generating Grad-CAM..."):
                gradcam_result = generate_gradcam(
                    frontal,
                    lateral,
                    indication,
                    pipeline_slug,
                    threshold,
                )
            render_gradcam_output(gradcam_result, top_k, pipeline_label)
        else:
            with st.spinner("Running imaging model..."):
                prediction_result = predict_case(
                    frontal,
                    lateral,
                    indication,
                    pipeline_slug,
                    threshold,
                )
            render_prediction_output(prediction_result, top_k, pipeline_label, indication)
    except Exception as exc:
        st.error(f"The model could not complete this run: {exc}")

render_app()
