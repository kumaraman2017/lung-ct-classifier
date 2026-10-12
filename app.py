"""Lung CT image classifier (Benign vs Malignant) - Streamlit GUI.

Runs locally with `streamlit run app.py` and on Streamlit Community Cloud.
Model: MobileNetV2 fine-tuned on the IQ-OTH/NCCD lung cancer CT dataset (PyTorch, CPU).
For education only - not a medical device.
"""
import json
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch import nn
from torchvision import models

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model" / "mobilenetv2_lung.pt"
ASSETS_DIR = APP_DIR / "assets"
SAMPLES_DIR = APP_DIR / "samples"

st.set_page_config(page_title="Lung CT Classifier", page_icon="🫁", layout="wide")


@st.cache_resource(show_spinner="Loading model ...")
def load_model():
    """Load the MobileNetV2 checkpoint once per server process."""
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.last_channel, len(ckpt["class_names"])))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@st.cache_data
def load_metrics():
    path = ASSETS_DIR / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


def to_rgb(image):
    """Decode fully, apply EXIF rotation, and convert to 8-bit RGB (16-bit and float
    images, e.g. CT slices exported from DICOM, are min-max scaled to 0-255 first)."""
    image.load()
    image = ImageOps.exif_transpose(image)
    if image.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        a = np.asarray(image, dtype=np.float32)
        a = (a - a.min()) / max(float(a.max() - a.min()), 1e-6) * 255.0
        image = Image.fromarray(a.astype(np.uint8))
    return image.convert("RGB")


def preprocess(image, ckpt):
    """Same steps as evaluation in training: RGB, 224x224 bilinear, scale to [0,1], ImageNet normalisation."""
    size = ckpt["img_size"]
    x = np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    x = (x - np.asarray(ckpt["mean"], dtype=np.float32)) / np.asarray(ckpt["std"], dtype=np.float32)
    return torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0)


def predict(image):
    model, ckpt = load_model()
    with torch.inference_mode():
        probs = torch.softmax(model(preprocess(image, ckpt)), dim=1)[0].tolist()
    return dict(zip(ckpt["class_names"], probs))


def classify_tab():
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("1. Upload a CT slice")
        uploaded = st.file_uploader("Choose a lung CT image (JPG or PNG)", type=["jpg", "jpeg", "png"])
        samples = {p.stem.replace("_", " ").title(): p for p in sorted(SAMPLES_DIR.glob("*.jpg"))}
        choice = st.selectbox("...or try a sample image from the test set", ["None", *samples])
        source = uploaded if uploaded is not None else samples.get(choice)
        if uploaded is not None and choice != "None":
            st.caption("Using your uploaded file. Remove it (x) to try a sample.")
        if source is None:
            st.info("Upload an image or pick a sample to get a prediction.")
            return
        try:
            image = to_rgb(Image.open(source))
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
            st.error("This file could not be read as an image. Please upload a JPG or PNG file.")
            return
        st.image(image, caption="Image preview", width="stretch")

    with right:
        st.subheader("2. Prediction")
        probs = predict(image)
        label = max(probs, key=probs.get)
        confidence = probs[label]
        box = st.error if label == "Malignant" else st.success
        box(f"Predicted class: **{label}**")
        st.metric("Confidence", f"{confidence:.1%}")
        for name, p in probs.items():
            st.progress(p, text=f"{name}: {p:.1%}")
        st.caption("Decision rule: Malignant if P(Malignant) >= 0.5, otherwise Benign.")
        if confidence < 0.7:
            st.warning("Low confidence: the model is unsure about this image.")
        st.caption(
            "The model was trained only on axial chest CT slices from the IQ-OTH/NCCD dataset. "
            "Other kinds of images give meaningless results."
        )


def performance_tab():
    metrics = load_metrics()
    if metrics is None:
        st.warning("metrics.json not found in assets/.")
        return
    st.subheader("Test-set metrics")
    rows = []
    for m in metrics["models"].values():
        t = m["test"]
        rows.append({
            "Model": m["display_name"],
            "Accuracy": f"{t['accuracy']:.3f}",
            "Precision": f"{t['precision']:.3f}",
            "Recall (sensitivity)": f"{t['recall']:.3f}",
            "Specificity": f"{t['specificity']:.3f}",
            "F1-score": f"{t['f1']:.3f}",
            "ROC AUC": f"{t['auc']:.3f}",
        })
    st.dataframe(rows, hide_index=True, width="stretch")
    split = metrics["dataset"]["split_counts"]
    st.caption(
        "Positive class = Malignant. Test set: "
        f"{split['test']['Benign']} benign + {split['test']['Malignant']} malignant images, "
        "kept separate from training by a group-aware split. The deployed model is MobileNetV2."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.image(str(ASSETS_DIR / "confusion_matrix_mobilenetv2.png"), caption="Confusion matrix (MobileNetV2, test set)")
    with c2:
        st.image(str(ASSETS_DIR / "roc_curves.png"), caption="ROC curves (test set)")
    with st.expander("Training curves (MobileNetV2)"):
        st.image(str(ASSETS_DIR / "training_curves_mobilenetv2.png"), width="stretch")


def about_tab():
    st.markdown(
        """
**What this app does.** It classifies a single lung CT slice as **Benign** or **Malignant**
using a MobileNetV2 network (ImageNet weights, fine-tuned with PyTorch).

**Data.** IQ-OTH/NCCD lung cancer dataset (Alyasriy and Al-Huseiny, Mendeley Data,
doi:10.17632/bhmdr45bh2.4, CC BY 4.0). Only the Benign and Malignant classes are used.

**Hosting.** Streamlit Community Cloud runs this app in a container. The code and model are
stored in a public GitHub repository.

**Team.** Aman Kumar (BTECH/10866/23), Mayank Kumar Agarwal (BTECH/10881/23),
Divyanshu Kumar (BTECH/10923/23), Cloud Computing Lab, Semester 7,
Birla Institute of Technology, Mesra, Ranchi.
"""
    )


st.title("🫁 Lung CT Classifier: Benign vs Malignant")
st.warning(
    "**For education only.** This is a student project, not a medical device. "
    "Do not use it for diagnosis or treatment decisions. Always consult a qualified doctor."
)
tab1, tab2, tab3 = st.tabs(["🔍 Classify", "📊 Model Performance", "ℹ️ About"])
with tab1:
    classify_tab()
with tab2:
    performance_tab()
with tab3:
    about_tab()
