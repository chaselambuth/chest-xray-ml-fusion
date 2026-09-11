# Grad-CAM generation and visualization helpers.

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# Capture activations and gradients from a target layer for Grad-CAM maps.
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    # Store forward activations from the hooked layer.
    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    # Store gradients flowing through the hooked layer.
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    # Generate a normalized class activation map and detached logits.
    def generate(self, frontal, lateral, input_ids, attention_mask, target_class=None, size: int = 224):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(frontal, lateral, input_ids, attention_mask)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
        logits[:, target_class].sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(size, size), mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam, logits.detach()

    # Remove registered hooks after Grad-CAM generation.
    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


# Blend a Grad-CAM heatmap onto a PIL image.
def overlay_gradcam(image: Image.Image, cam: np.ndarray, alpha: float = 0.45, size: int = 224) -> Image.Image:
    base = image.resize((size, size)).convert("RGB")
    base_arr = np.array(base, dtype=np.float32) / 255.0
    heat = plt.get_cmap("jet")(cam)[..., :3]
    overlay = (1 - alpha) * base_arr + alpha * heat
    return Image.fromarray((np.clip(overlay, 0.0, 1.0) * 255).astype(np.uint8))

# Convert a CHW image tensor into a clipped HWC numpy array.
def tensor_to_image(tensor) -> np.ndarray:
    array = tensor.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(array, 0.0, 1.0)

# Blend a heatmap with a normalized RGB image array.
def overlay_heatmap(image_array: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = plt.get_cmap("jet")(cam)[..., :3]
    return np.clip((1 - alpha) * image_array + alpha * heat, 0.0, 1.0)

# Save a three-panel original, heatmap, and overlay Grad-CAM figure.
def save_cam_panel(image_array: np.ndarray, cam: np.ndarray, overlay: np.ndarray, save_path, title: str) -> None:
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
