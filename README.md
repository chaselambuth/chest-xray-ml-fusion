# Chest X-ray ML Fusion

Machine learning research project testing whether multimodal chest X-ray data, combining dual-view images with radiology report text, improves classification performance over image-only models. Follow-up experiments compare text encoders, residual gated fusion, segmentation preprocessing, tuning, and Grad-CAM interpretability.

This project is for research and educational use only. It is not a clinical diagnostic system and should not be used for medical decision-making.

This folder is the GitHub-oriented version of the local research workspace. It keeps the current source code, stripped notebooks, lightweight result summaries, and the final case analysis while leaving out obsolete archives, legacy outputs, caches, and generated image/checkpoint-heavy artifacts.

## Project Summary

The primary experiment asks whether multimodal machine learning can improve five-class chest X-ray classification beyond image-only baselines. The strongest supported direction is a Bio_ClinicalBERT residual gated fusion model using the original full frontal and lateral X-ray images. Fusion helped compared with the starting baselines, but the model remains limited by class imbalance and weak minority-class recall.

Final-stage held-out results:

| Model / Pipeline | Test Accuracy | Test Macro F1 | Notes |
| --- | ---: | ---: | --- |
| Original full-image Bio_ClinicalBERT residual gated fusion | 0.737 | 0.420 | Retained final checkpoint |
| Tuned intensity-jitter regularized checkpoint | 0.674 | 0.408 | Better validation macro F1, weaker test generalization |
| Lung-cropped frontal plus original lateral | 0.765 | 0.384 | Higher accuracy, weaker balanced performance |
| Lung-masked frontal and lateral | 0.286 | 0.194 | Collapsed under masked preprocessing |

The case analysis is available at [case_analysis.md](case_analysis.md).

## Data Source

This project uses the Indiana University Chest X-ray Collection from Open-i, distributed by the U.S. National Library of Medicine. The local data files follow the common Kaggle mirror layout:

- `indiana_projections.csv`
- `indiana_reports.csv`
- `images_normalized/`

Dataset links:

- Open-i / NLM: <https://openi.nlm.nih.gov/>
- Open-i FAQ: <https://openi-vip.nlm.nih.gov/faq>
- Kaggle mirror: <https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university>

Suggested dataset citation:

> Demner-Fushman D, Kohli MD, Rosenman MB, Shooshan SE, Rodriguez L, Antani S, Thoma GR, McDonald CJ. Preparing a collection of radiology examinations for distribution and retrieval. Journal of the American Medical Informatics Association. 2016;23(2):304-310.

The public repository should not commit raw images, local caches, generated heatmaps, or model checkpoints. For local Docker testing from this folder, `compose.yaml` mounts those artifacts from the parent research workspace.

## Notebook Workflow

| Notebook | Role | Main Question |
| --- | --- | --- |
| `experiment_notebooks/01_Xray_Baseline.ipynb` | Dataset creation and first baselines | Do image, text, or simple fusion features provide the strongest starting point? |
| `experiment_notebooks/02_Xray_Text_Encoder_Selection.ipynb` | Clinical text encoder comparison | Which text encoder should be carried forward into gated fusion? |
| `experiment_notebooks/03_Xray_ClinicalBERT_Gated_Fusion_Test.ipynb` | Residual gated fusion ablation | Does gated image/text interaction improve held-out performance? |
| `experiment_notebooks/04_Xray_Segmentation_Test.ipynb` | Lung segmentation preprocessing test | Should the model use full images, cropped images, or masked images? |
| `experiment_notebooks/05_Xray_Hyperparameter_Augmentation.ipynb` | Hyperparameter and augmentation tuning | Does conservative tuning improve validation and held-out test performance? |
| `experiment_notebooks/06_Xray_GradCAM_Interpretability.ipynb` | Final Grad-CAM interpretation | Do the selected model's case-level heatmaps support the metric findings? |

Notebook outputs are stripped for source control. Lightweight CSV/JSON summaries are retained in `notebooks_outputs/`.

## Main Findings

- Image features are the backbone of performance; text-only models were too weak for the five-class task.
- Multimodal fusion helped, and residual gated fusion improved the carried-forward Bio_ClinicalBERT fusion model.
- Bio_ClinicalBERT was selected because it produced the strongest held-out fusion accuracy and provided a stable clinical-language backbone.
- Lung segmentation preprocessing did not become the final pipeline; full images produced stronger balanced behavior.
- Hyperparameter and augmentation tuning improved validation macro F1, but did not improve held-out test performance.
- Grad-CAM helped connect metrics to case-level behavior, but many maps were blank or weak.
- The dominant failure pattern is abnormal studies being predicted as normal.

## Repository Layout

```text
.
|-- experiment_notebooks/              # stripped notebooks 01-06
|-- notebooks_outputs/                 # lightweight CSV/JSON result summaries
|-- src/
|   `-- xray_fusion/                   # shared experiment utilities
|-- static/
|-- templates/
|-- tools/                             # active notebook-building helpers
|-- app.py                             # Streamlit app entrypoint
|-- flask_app.py                       # Flask app entrypoint
|-- xray_inference.py                  # shared inference helper
|-- case_analysis.md
|-- Dockerfile
|-- compose.yaml
|-- requirements.txt
|-- .gitignore
`-- .dockerignore
```

## Files Excluded From This Version

The GitHub version intentionally excludes:

- archive folders
- legacy experiment outputs
- notebook checkpoints
- Python and pytest caches
- Hugging Face caches
- raw/normalized image data
- split CSVs and derived dataset CSVs
- model checkpoints and weights
- generated Grad-CAM heatmap PNGs

Those local artifacts can stay in the parent research workspace while Docker is being tested.

## Environment

The project currently uses Python 3.11 and the dependencies listed in `requirements.txt`.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Testing

Run the lightweight pytest suite:

```bash
pytest
```

Use the repository check before pushing:

```bash
python tools/prepush_check.py
```

GPU acceleration is recommended for training and rerunning the notebooks. The notebooks are preserved as source notebooks, not as a fast end-to-end rerun package.

## Running The Apps

### Streamlit

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

### Flask

```bash
py flask_app.py
```

Then open:

```text
http://localhost:5001
```

The apps require local checkpoints and model/cache artifacts for checkpoint-backed inference.

## Docker

From this `github_version` folder:

```bash
docker compose up --build
```

Then open:

```text
http://localhost:8501
```

For local testing, `compose.yaml` mounts these parent-workspace folders into the container:

- `../hf_cache` -> `/app/hf_cache`
- `../splits` -> `/app/splits`
- `../Experiments/outputs/experiment_04_segmentation` -> `/app/Experiments/outputs/experiment_04_segmentation`

If this folder is cloned somewhere else, recreate those local folders or update the volume paths before running Docker.

## Current Limitations

- This is not a clinical diagnostic system.
- The five-class label space is simplified and can hide label ambiguity in the original reports.
- The dataset is strongly imbalanced toward normal studies.
- Minority-class recall remains unstable.
- Grad-CAM outputs are qualitative failure-analysis artifacts, not proof of clinical localization.
- A full reproducibility pass should add `DATA.md`, `LICENSE`, and `CITATION.cff`.
