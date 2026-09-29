# Automated Accessibility Evaluation of News Websites

A Streamlit web application that estimates the **accessibility level** (High / Medium / Low) of a news webpage. It combines screenshot-based visual features, DOM/HTML-based features, and (optionally) deep-learning visual embeddings, then explains each prediction with **SHAP** and **Grad-CAM**.

---

## Table of Contents

1. [What the app does](#1-what-the-app-does)
2. [Prerequisites](#2-prerequisites)
3. [Project structure](#3-project-structure)
4. [Step-by-step installation](#4-step-by-step-installation)
5. [Add the trained model files (required)](#5-add-the-trained-model-files-required)
6. [Run the application](#6-run-the-application)
7. [Using the app](#7-using-the-app)
8. [Understanding the results](#8-understanding-the-results)
9. [Notes and limitations](#9-notes-and-limitations)

---

## 1. What the app does

You give the app a webpage (either by browsing to it live, or by uploading a screenshot plus its HTML). The app then:

1. Extracts **visual features** from the screenshot (edge density, contrast, colour variance, whitespace ratio, etc.).
2. Extracts **DOM features** from the HTML (number of links/images/headings, ARIA attributes, missing alt text, heading-level skips, landmarks, unlabelled buttons/links/inputs, etc.).
3. Predicts an accessibility class: **High**, **Medium**, or **Low** Accessibility, using either:
   - **Tabular only**: Random Forest or Gradient Boosting on the numeric features, or
   - **Hybrid**: a CNN/ViT visual embedding of the screenshot combined with the numeric features.
4. Explains the prediction:
   - **SHAP**: which features pushed the prediction toward or away from the result.
   - **Grad-CAM / Grad-CAM++ / Layer-CAM**: which regions of the screenshot the visual model focused on, mapped back to page elements.

Supported visual architectures: **ResNet18, ResNet50, VGG16, ViT-B/16, Swin-T**.

---

## 2. Prerequisites

Install these **before** you begin:

| Requirement | Details |
|---|---|
| **Python** | 3.10 recommended (the project was developed with Python 3.10). 3.9 to 3.11 should also work. |
| **pip** | Comes with Python. Upgrade with `python -m pip install --upgrade pip`. |
| **Google Chrome** | Required. The app uses Selenium to drive Chrome. [Download Chrome](https://www.google.com/chrome/). |
| **Internet connection** | Needed on first run so `webdriver-manager` can download a matching ChromeDriver, and so pretrained PyTorch weights can be fetched if needed. |
| **A desktop environment** | The *Interactive Browser* mode opens a **visible** Chrome window, so run the app on your own computer (not on a headless server or a hosted platform such as Streamlit Community Cloud). |
| **Git** | Optional, for cloning the repository. |
| **Trained model files** | Required. See [Section 5](#5-add-the-trained-model-files-required). |

> **GPU:** Optional. If a CUDA-capable GPU and a CUDA build of PyTorch are available, hybrid mode will use it automatically; otherwise it runs on CPU.

---

## 3. Project structure

```
your-repo/
├── app.py                      # Streamlit entry point
├── requirements.txt            # Python dependencies (see Section 4)
├── README.md
├── models/                     # NOT included in the zip; you must add it (Section 5)
│   ├── rf_model.pkl
│   ├── gb_model.pkl
│   ├── scaler.pkl
│   ├── resnet18_model.pth
│   ├── resnet50_model.pth
│   ├── vgg16_model.pth
│   ├── vit_b_16_model.pth
│   ├── swin_t_model.pth
│   └── hybrid_models/
│       ├── enhanced_<arch>_<head>_model.pkl
│       ├── enhanced_<arch>_<head>_embedding_scaler.pkl
│       ├── enhanced_<arch>_<head>_tabular_scaler.pkl
│       └── enhanced_<arch>_<head>_tabular_columns.pkl
├── live_outputs/               # Created/used at runtime for screenshots, HTML, element maps
└── utils/
    ├── explainability.py       # SHAP explanations
    ├── gradcam.py              # Visual model loading + Grad-CAM generation
    ├── gradcam_explain.py      # Turns heatmaps into hotspot regions / plain-English text
    ├── hybrid_prediction.py    # Visual embedding + tabular hybrid model
    ├── live_capture.py         # Selenium browser control, DOM element capture
    ├── live_features.py        # Screenshot and DOM feature extraction
    └── live_prediction.py      # Tabular-only prediction (RF / GB)
```

---

## 4. Step-by-step installation

### Step 1: Get the code

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

If you downloaded a ZIP instead, unzip it and open a terminal inside the extracted folder (the one containing `app.py`).

### Step 2: Create a virtual environment (recommended)

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```
If PowerShell blocks the script, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, or use Command Prompt and `venv\Scripts\activate.bat`.

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should now see `(venv)` at the start of your terminal prompt.

### Step 3: Install dependencies

Create a file named `requirements.txt` in the project root with the following contents (if the repo does not already include one):

```txt
streamlit>=1.36
pandas
numpy
pillow
requests
matplotlib
scikit-learn
joblib
shap
num2words
beautifulsoup4
opencv-python
selenium
webdriver-manager
torch
torchvision
timm
grad-cam
```

Then install:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Notes:

- The PyPI package is called **`grad-cam`**, but it is imported in code as `pytorch_grad_cam`.
- `torch` and `torchvision` are large downloads (several hundred MB). For a specific CUDA/CPU build, use the selector at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).
- **Important:** the `.pkl` model files are scikit-learn pickles. Install the **same scikit-learn version used to train them**, otherwise loading may fail or warn. If you know it, pin it, e.g. `scikit-learn==1.4.2`.
- Streamlit **1.36 or newer** is recommended because the app uses `st.html` and `use_container_width`.

### Step 4: Verify the installation

```bash
python -c "import streamlit, torch, torchvision, timm, shap, cv2, selenium, pytorch_grad_cam; print('All imports OK')"
```

If this prints `All imports OK`, you are ready to continue.

---

## 5. Add the trained model files (required)

The application **will not run predictions without trained models**. They are loaded from a `models/` folder located next to `app.py`. Download the `models/` folder using the provided Google Drive link at the root of the project repository and place it in this directory. Kindly ensure that the model files are using **exactly** these names:

### 5a. Tabular models (needed for every run)

| File | Purpose |
|---|---|
| `models/rf_model.pkl` | Random Forest classifier |
| `models/gb_model.pkl` | Gradient Boosting classifier |
| `models/scaler.pkl` | Feature scaler used at training time |

### 5b. Visual checkpoints (needed for Grad-CAM and Hybrid mode)

Each must be a PyTorch checkpoint with **3 output classes** matching the selected architecture:

| File | Architecture |
|---|---|
| `models/resnet18_model.pth` | ResNet-18 |
| `models/resnet50_model.pth` | ResNet-50 |
| `models/vgg16_model.pth` | VGG-16 |
| `models/vit_b_16_model.pth` | ViT-B/16 |
| `models/swin_t_model.pth` | Swin-T |

You only need the checkpoints for the architectures you plan to use. You can also point the app at a checkpoint stored elsewhere using the **Visual model checkpoint** box in the sidebar.

### 5c. Hybrid models (needed only for Hybrid mode)

Placed in `models/hybrid_models/`. For each combination of architecture and head, four files are required:

```
enhanced_{architecture}_{head}_model.pkl
enhanced_{architecture}_{head}_embedding_scaler.pkl
enhanced_{architecture}_{head}_tabular_scaler.pkl
enhanced_{architecture}_{head}_tabular_columns.pkl
```

Where:
- `{architecture}` is one of `resnet18`, `resnet50`, `vgg16`, `vit_b_16`, `swin_t`
- `{head}` is `rf` (Random Forest) or `gb` (Gradient Boosting)

Example for ResNet18 + Random Forest:
```
models/hybrid_models/enhanced_resnet18_rf_model.pkl
models/hybrid_models/enhanced_resnet18_rf_embedding_scaler.pkl
models/hybrid_models/enhanced_resnet18_rf_tabular_scaler.pkl
models/hybrid_models/enhanced_resnet18_rf_tabular_columns.pkl
```

> Model files are usually too large for a normal Git commit. If you host them in this repo, consider [Git LFS](https://git-lfs.com/), or host them elsewhere (Google Drive, Hugging Face, GitHub Releases) and add the download link here: **`<add your model download link>`**.

---

## 6. Run the application

Make sure your virtual environment is active and you are in the project root (where `app.py` lives), then run:

```bash
streamlit run app.py
```

Your browser should open it automatically. If not, paste the URL into your browser manually.

To stop the app, simply type **Ctrl + C** in the terminal.
---

## 7. Using the app

### Sidebar settings ("Evaluation settings")

| Setting | What it does |
|---|---|
| **Prediction model** | *Tabular only* uses DOM and screenshot-derived numeric features. *Hybrid* also feeds a CNN/ViT embedding of the screenshot into an enhanced model. |
| **Model head** | Choose *Random Forest* or *Gradient Boosting*. |
| **Generate SHAP explanation** | On by default. Shows feature contributions. |
| **Generate Grad-CAM explanation** | Off by default. Highlights influential regions of the screenshot. Requires a visual checkpoint. |
| **Visual architecture** | ResNet18, ResNet50, VGG16, ViT_B_16, or Swin-T. Enabled when Grad-CAM or Hybrid is on. |
| **Visual model checkpoint** | Path to the `.pth` file. Auto-fills based on the architecture chosen. |
| **Grad-CAM method** | Grad-CAM, Grad-CAM++, or Layer-CAM. |

### Input option A: Interactive Browser (Navigate manually)

1. Select **Interactive Browser (Navigate manually)**.
2. Enter a webpage URL (e.g. `https://example.com/news/article`). Adding `https://` is optional.
3. Click **1. Launch Browser Window**. A Chrome window opens on that page. The first launch may take a little longer while ChromeDriver is downloaded.
4. In that Chrome window, **scroll, close cookie banners/pop-ups, or navigate** to the exact view you want evaluated.
5. Return to the Streamlit tab and click **2. Capture Current View & Evaluate**.
6. The app screenshots the current view, saves the page HTML, records element positions, closes Chrome, and runs the analysis.

> The screenshot captures **what is currently visible** in the Chrome window, not the full scrolling page.

### Input option B: Upload Screenshot & HTML

1. Select **Upload Screenshot & HTML**.
2. Upload a screenshot (`.png`, `.jpg`, `.jpeg`).
3. Upload the matching HTML source (`.html` or `.txt`), e.g. saved via "Save Page As" or copied from DevTools.
4. Click **Evaluate uploaded files**.

> **Important:** The screenshot and HTML must come from the **same page at the same moment**. Mismatched files lead to inaccurate predictions.

For Grad-CAM explanations in this mode, the app re-renders your HTML offline in a headless browser to map page elements. This is approximate, especially if the page relies on relative-path CSS/JS/images. If it fails, prediction and SHAP still work.

---

## 8. Understanding the results

After an evaluation you will see:

- **Headline metrics**: predicted class, confidence, and number of features extracted.
- **Captured webpage** tab: the screenshot that was analysed.
- **Features and probabilities** tab: every extracted feature value and a bar chart of class probabilities.
- **SHAP** tab: a bar chart of feature contributions, plain-English explanations of factors pushing toward, against, or negligibly affecting the result, and an optional raw SHAP table. In Hybrid mode the visual embedding dimensions are summed into one "Visual embedding (aggregate)" row for readability.
- **Grad-CAM** tab: the original screenshot beside the heatmap overlay, the visual model's own prediction and confidence, a text description of the hotspot regions (with matched DOM elements when available), and an optional debug overlay.

Important to know:
- In **Hybrid** mode, the headline prediction comes from the hybrid model, while Grad-CAM explains the standalone visual checkpoint. The two can legitimately disagree.
- SHAP explains the tabular/hybrid classifier. Grad-CAM explains the visual model. They are separate explanations.

Intermediate files (screenshots, saved HTML, element maps) are written to the `live_outputs/` folder and overwritten on each run.

---

## 9. Notes and limitations

- Intended for **public news webpages**; results depend on the quality of the captured screenshot and HTML.
- The **Interactive Browser** mode requires a local desktop session with Chrome installed.
- Predictions are automated estimates and do **not** replace a full manual accessibility audit (for example, against WCAG guidelines).
