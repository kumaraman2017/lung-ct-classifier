# Lung CT Classifier (Benign vs Malignant)

A Streamlit web app that classifies a lung CT slice as **Benign** or **Malignant** with a
MobileNetV2 model fine-tuned in PyTorch. It is deployed on **Streamlit Community Cloud**.

> **For education only.** This is a student project, not a medical device. Do not use it for diagnosis.

**Live app:** [https://lung-ct-classifier-ovxxsy7qdputbfhspwzmwk.streamlit.app/](https://lung-ct-classifier-ovxxsy7qdputbfhspwzmwk.streamlit.app/)

## Files

| Path | Purpose |
|---|---|
| `app.py` | Streamlit GUI: upload, preview, prediction with confidence, Model Performance tab |
| `requirements.txt` | Pinned packages (CPU-only PyTorch wheels via `--find-links`) |
| `model/mobilenetv2_lung.pt` | Trained MobileNetV2 weights (PyTorch `state_dict`, about 9 MB) |
| `assets/` | Confusion matrix, ROC curves, training curves and `metrics.json` used by the app |
| `samples/` | One benign and one malignant test image to try the app |

## Run locally

Needs Python 3.12 or newer (on macOS: Apple Silicon with macOS 14+).

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> → **Continue to sign-in** → **Continue with GitHub**.
2. Click **Create app** → **Yup, I have an app**.
3. Repository: this repo, Branch: `main`, Main file path: `app.py`.
4. Open **Advanced settings** → Python version **3.12** → **Save**.
5. Click **Deploy** and wait for the first build (a few minutes, because PyTorch is installed).

## Dataset and credit

IQ-OTH/NCCD lung cancer dataset: H. Alyasriy and M. Al-Huseiny, Mendeley Data, V4, 2023,
doi:[10.17632/bhmdr45bh2.4](https://doi.org/10.17632/bhmdr45bh2.4), licence CC BY 4.0.
Only the Benign and Malignant classes are used.

## Team

Aman Kumar (BTECH/10866/23), Mayank Kumar Agarwal (BTECH/10881/23), Divyanshu Kumar (BTECH/10923/23).
Cloud Computing Lab, Semester 7, Birla Institute of Technology, Mesra, Ranchi.
