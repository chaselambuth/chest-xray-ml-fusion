# Small image helpers for lung-mask preprocessing and overlays.

from __future__ import annotations

import numpy as np
from PIL import Image


# Return the tight bounding box around nonzero mask pixels, if any.
def largest_bbox_from_mask(mask_array: np.ndarray):
    ys, xs = np.where(mask_array > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

# Apply a binary mask to a PIL image and return RGB output.
def masked_image(image: Image.Image, mask_array: np.ndarray) -> Image.Image:
    image_arr = np.array(image.convert("RGB"), dtype=np.uint8)
    mask01 = (mask_array > 0).astype(np.uint8)
    return Image.fromarray(image_arr * mask01[..., None])

# Crop a frontal image to the mask box, falling back to the full image.
def cropped_frontal_image(image: Image.Image, mask_array: np.ndarray) -> Image.Image:
    bbox = largest_bbox_from_mask(mask_array)
    if bbox is None:
        return image.convert("RGB")
    return image.convert("RGB").crop(bbox)


# Load an image path and apply a binary mask as an RGB array.
def masked_rgb_array(image_path, mask_array: np.ndarray) -> np.ndarray:
    image_arr = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    mask01 = (mask_array > 0).astype(np.uint8)
    return image_arr * mask01[..., None]

# Load and crop an image path using a lung mask bounding box.
def cropped_frontal_array(image_path, mask_array: np.ndarray) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    bbox = largest_bbox_from_mask(mask_array)
    if bbox is None:
        return np.array(image, dtype=np.uint8)
    return np.array(image.crop(bbox), dtype=np.uint8)


# Blend a teal binary mask overlay onto a PIL image.
def overlay_mask(image: Image.Image, mask_array: np.ndarray, alpha: float = 0.38) -> Image.Image:
    image_arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    mask = (mask_array > 0).astype(np.float32)
    color = np.zeros_like(image_arr)
    color[..., 0] = 0.08
    color[..., 1] = 0.62
    color[..., 2] = 0.67
    overlay = image_arr * (1 - alpha * mask[..., None]) + color * (alpha * mask[..., None])
    return Image.fromarray((np.clip(overlay, 0.0, 1.0) * 255).astype(np.uint8))
