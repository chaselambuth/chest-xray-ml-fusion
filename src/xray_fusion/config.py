# Shared project constants and path helpers for notebook and app code.

from __future__ import annotations

from pathlib import Path


DEFAULT_IMAGE_SIZE = 224
DEFAULT_SEGMENTATION_IMAGE_SIZE = 512
DEFAULT_MAX_LEN = 64

DEFAULT_LABEL_DISPLAY = {
    "cardiomegaly": "Cardiomegaly",
    "chronic_lung_disease": "Chronic Lung Disease",
    "normal": "Normal",
    "pleural_effusion": "Pleural Effusion",
    "pneumonia_or_opacity": "Pneumonia or Opacity",
}


# Find the repository-like root by looking for expected data/output folders.
def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "splits").exists() and (candidate / "Experiments").exists():
            return candidate
    return current


# Add the local src directory to sys.path for notebook execution.
def add_src_to_path(project_root: Path) -> None:
    import sys

    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.append(str(src_path))
