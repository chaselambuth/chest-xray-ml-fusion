# Flask entrypoint for the local XRAY Fusion web interface.

from __future__ import annotations

import base64
import io

from flask import Flask, render_template, request
from PIL import Image

import xray_inference as xray

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024


# Convert an uploaded file into an RGB PIL image, if present.
def image_from_upload(uploaded_file):
    if uploaded_file is None or not uploaded_file.filename:
        return None
    return Image.open(uploaded_file.stream).convert("RGB")

# Encode a PIL image as a PNG data URI for embedding in templates.
def image_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"

# Package an image and caption for the Jinja template.
def image_payload(image: Image.Image, caption: str):
    return {"src": image_data_uri(image), "caption": caption}


# Read and sanitize the form controls from the current request.
def request_state():
    mode = request.form.get("mode", "Prediction")
    if mode not in {"Prediction", "Segmentation"}:
        mode = "Prediction"
    pipeline_slug = request.form.get("pipeline_slug", "b_lung_cropped_frontal")
    top_k = int(request.form.get("top_k", 5))
    threshold = float(request.form.get("threshold", 0.5))
    indication = request.form.get("indication", "")
    return mode, pipeline_slug, top_k, threshold, indication

# Build the base template context with optional result data.
def render_context(**kwargs):
    return {
        "pipeline_specs": xray.PIPELINE_SPECS,
        "selected_mode": kwargs.pop("selected_mode", "Prediction"),
        "selected_pipeline": kwargs.pop("selected_pipeline", "b_lung_cropped_frontal"),
        "selected_top_k": kwargs.pop("selected_top_k", 5),
        "selected_threshold": kwargs.pop("selected_threshold", 0.5),
        "indication": kwargs.pop("indication", ""),
        **kwargs,
    }


@app.route("/", methods=["GET", "POST"])
# Render the upload form and run prediction or segmentation on POST.
def index():
    if request.method == "GET":
        return render_template("index.html", **render_context())

    mode, pipeline_slug, top_k, threshold, indication = request_state()
    context = render_context(
        selected_mode=mode,
        selected_pipeline=pipeline_slug,
        selected_top_k=top_k,
        selected_threshold=threshold,
        indication=indication,
    )

    frontal = image_from_upload(request.files.get("frontal"))
    lateral = image_from_upload(request.files.get("lateral"))

    if frontal is None or lateral is None:
        context["error"] = "Please upload both frontal and lateral chest X-ray images."
        return render_template("index.html", **context), 400

    try:
        context["uploads"] = [
            image_payload(frontal, "Frontal upload"),
            image_payload(lateral, "Lateral upload"),
        ]

        if mode == "Segmentation":
            result = xray.run_segmentation(frontal, lateral, threshold)
            context["segmentation_images"] = [
                image_payload(frontal, "Frontal original"),
                image_payload(result["frontal_mask"], "Frontal mask"),
                image_payload(result["frontal_overlay"], "Frontal overlay"),
                image_payload(lateral, "Lateral original"),
                image_payload(result["lateral_mask"], "Lateral mask"),
                image_payload(result["lateral_overlay"], "Lateral overlay"),
            ]
        else:
            result = xray.generate_gradcam(frontal, lateral, indication, pipeline_slug, threshold)
            predictions = xray.top_k_predictions(result["label_names"], result["probs"], top_k)
            top_prediction = predictions[0]
            context["predictions"] = predictions
            context["top_prediction"] = top_prediction
            context["model_images"] = [
                image_payload(result["model_frontal"], "Frontal analysis view"),
                image_payload(result["model_lateral"], "Lateral analysis view"),
            ]
            context["attention_images"] = [
                image_payload(result["model_frontal"].resize((xray.IMG_SIZE, xray.IMG_SIZE)), "Frontal analysis view"),
                image_payload(result["overlay"], "Attention overlay"),
            ]
            context["report"] = xray.make_report(
                top_prediction["label"],
                top_prediction["score"],
                xray.PIPELINE_SPECS[pipeline_slug]["name"],
                indication,
            )
    except Exception as exc:
        context["error"] = str(exc)
        return render_template("index.html", **context), 500

    return render_template("index.html", **context)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
