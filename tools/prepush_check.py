# Lightweight checks before pushing the GitHub version.

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
    "archive",
    "legacy",
    "hf_cache",
    "resolved_hf_models",
    "images_normalized",
    "splits",
    "notebook_originals_before_python_refactor_2026_08_18",
}

FORBIDDEN_FILES = {
    "indiana_projections.csv",
    "indiana_reports.csv",
    "study_level_dataset.csv",
    "flask_command.txt",
    "streamlit_command.txt",
}

FORBIDDEN_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".safetensors",
}

MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024


# Return repository paths while skipping Git metadata.
def iter_paths() -> list[Path]:
    return [path for path in ROOT.rglob("*") if ".git" not in path.parts]


# Find local-only artifacts that should not be pushed.
def check_forbidden_artifacts(paths: list[Path]) -> list[str]:
    issues: list[str] = []
    for path in paths:
        rel = path.relative_to(ROOT)
        if path.is_dir() and path.name in FORBIDDEN_DIRS:
            issues.append(f"Forbidden directory: {rel}")
        if path.is_file() and path.name in FORBIDDEN_FILES:
            issues.append(f"Forbidden file: {rel}")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            issues.append(f"Forbidden model artifact: {rel}")
        if path.is_file() and path.stat().st_size > MAX_FILE_SIZE_BYTES:
            issues.append(f"Large file over 100 MB: {rel}")
    return issues


# Verify notebooks are valid notebook files.
def check_notebooks_parse() -> list[str]:
    try:
        import nbformat
    except ImportError:
        return ["nbformat is not installed, so notebook validation could not run."]

    issues: list[str] = []
    for notebook_path in (ROOT / "experiment_notebooks").glob("*.ipynb"):
        try:
            nbformat.read(notebook_path, as_version=4)
        except Exception as exc:  # pragma: no cover - validation helper
            rel = notebook_path.relative_to(ROOT)
            issues.append(f"Notebook did not parse: {rel} ({exc})")
    return issues


# Run all GitHub-version checks.
def main() -> int:
    paths = iter_paths()
    issues = check_forbidden_artifacts(paths)
    issues.extend(check_notebooks_parse())

    if issues:
        print("Pre-push check failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("Pre-push check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
