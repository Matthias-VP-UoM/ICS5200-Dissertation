from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

from utils.gradcam import (
    DEFAULT_IMAGE_SIZE,
    build_model,
    normalise_architecture_name,
    preprocess,
    load_visual_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
HYBRID_MODEL_DIR = MODEL_DIR / "hybrid_models"

EMBEDDING_FIRST = True
DEFAULT_USE_FINETUNED_BACKBONE = True

# Defines the number of dimensions each deep learning model should use
EMBEDDING_DIMS = {
    "resnet18": 512,
    "resnet50": 2048,
    "vgg16": 4096,
    "vit_b_16": 768,
    "swin_t": 768,
}

# Maps the sidebar-facing architecture label to the filename fragment used by the model filename
ARCHITECTURE_KEYS = {
    "ResNet18": "resnet18",
    "ResNet50": "resnet50",
    "VGG16": "vgg16",
    "ViT_B_16": "vit_b_16",
    "Swin-T": "swin_t",
}

# Defines the keys for the classical models
HEAD_KEYS = {
    "Random Forest": "rf",
    "Gradient Boosting": "gb",
}

# Defines the default path within which the deep learning models are stored
# NOTE: This can be configured by the user during application runtime
DEFAULT_CHECKPOINT = {
    "resnet18": MODEL_DIR / "resnet18_model.pth",
    "resnet50": MODEL_DIR / "resnet50_model.pth",
    "vgg16": MODEL_DIR / "vgg16_model.pth",
    "vit_b_16": MODEL_DIR / "vit_b_16_model.pth",
    "swin_t": MODEL_DIR / "swin_t_model.pth",
}


def strip_head(model: nn.Module, architecture: str) -> nn.Module:
    architecture = normalise_architecture_name(architecture)
    if architecture in {"resnet18", "resnet50"}:
        model.fc = nn.Identity()
    elif architecture == "vgg16":
        model.classifier[6] = nn.Identity()
    elif architecture == "vit_b_16":
        model.head = nn.Identity()
    elif architecture == "swin_t":
        model.head = nn.Identity()
    else:
        raise ValueError(f"Unsupported embedding architecture: {architecture}")
    return model

# Loads the selected deep learning visual model for embedding extraction
@lru_cache(maxsize=10)
def load_embedding_model(architecture: str, checkpoint_path: str | None, device_str: str):
    device = torch.device(device_str)
    norm_architecture = normalise_architecture_name(architecture)

    if checkpoint_path:
        model, _ = load_visual_model(checkpoint_path, norm_architecture, device)
    else:
        model, _ = build_model(norm_architecture, force_sequential_head=False)
        model.to(device).eval()

    model = strip_head(model, norm_architecture)
    model.to(device).eval()
    return model

# Runs the visual backbone on the screenshot and returns the resulting embedding
def extract_embedding(screenshot_path: str, architecture: str, checkpoint_path: str | Path | None, device: torch.device | None = None,) -> np.ndarray:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm_architecture = normalise_architecture_name(architecture)

    checkpoint_key = str(checkpoint_path) if checkpoint_path else None
    embedding_model = load_embedding_model(norm_architecture, checkpoint_key, str(device))

    image = Image.open(screenshot_path).convert("RGB")
    input_tensor = preprocess(image, DEFAULT_IMAGE_SIZE).to(device)

    with torch.no_grad():
        embedding = embedding_model(input_tensor)
        if embedding.ndim > 2:
            embedding = torch.flatten(embedding, start_dim=1)

    embedding_np = embedding.squeeze(0).cpu().numpy().astype(float)

    expected_dim = EMBEDDING_DIMS[norm_architecture]
    if embedding_np.shape[0] != expected_dim:
        raise ValueError(
            f"Extracted embedding for {architecture} has {embedding_np.shape[0]} dims, "
            f"expected {expected_dim}. The backbone/head-stripping logic may not match "
            "how the hybrid model was trained."
        )
    return embedding_np

# Loads the classical model + both scalers + tabular column order for one architecture/head combination, e.g. ('resnet18', 'gb')
@lru_cache(maxsize=10)
def load_hybrid_artifacts(architecture_key: str, head_key: str):
    base = HYBRID_MODEL_DIR / f"enhanced_{architecture_key}_{head_key}"
    model = joblib.load(f"{base}_model.pkl")
    embedding_scaler = joblib.load(f"{base}_embedding_scaler.pkl")
    tabular_scaler = joblib.load(f"{base}_tabular_scaler.pkl")
    tabular_columns = list(joblib.load(f"{base}_tabular_columns.pkl"))
    return model, embedding_scaler, tabular_scaler, tabular_columns

# Extracts the visual embedding, scales both feature blocks, concatenates them, and runs the classical model
def predict_hybrid(screenshot_path: str, tabular_features: dict, architecture: str, head_type: str = "Random Forest", checkpoint_path: str | Path | None = None, use_finetuned_backbone: bool = DEFAULT_USE_FINETUNED_BACKBONE, label_map: dict | None = None,):
    architecture_key = ARCHITECTURE_KEYS.get(architecture, normalise_architecture_name(architecture))
    head_key = HEAD_KEYS.get(head_type, head_type.lower())

    model, embedding_scaler, tabular_scaler, tabular_columns = load_hybrid_artifacts(
        architecture_key, head_key
    )

    effective_checkpoint = checkpoint_path if use_finetuned_backbone else None
    embedding_raw = extract_embedding(screenshot_path, architecture_key, effective_checkpoint)
    embedding_scaled = embedding_scaler.transform(embedding_raw.reshape(1, -1))[0]

    tabular_df = pd.DataFrame([tabular_features])
    missing = [c for c in tabular_columns if c not in tabular_df.columns]
    for c in missing:
        tabular_df[c] = 0.0
    tabular_raw = tabular_df[tabular_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    tabular_scaled = tabular_scaler.transform(tabular_raw.values)[0]

    embedding_columns = [f"emb_{i}" for i in range(embedding_scaled.shape[0])]
    if EMBEDDING_FIRST:
        combined_values = np.concatenate([embedding_scaled, tabular_scaled])
        combined_columns = embedding_columns + tabular_columns
    else:
        combined_values = np.concatenate([tabular_scaled, embedding_scaled])
        combined_columns = tabular_columns + embedding_columns

    X_scaled = pd.DataFrame([combined_values], columns=combined_columns)

    preds = model.predict(X_scaled.values)
    probs = model.predict_proba(X_scaled.values)[0]
    prediction = int(preds[0])
    confidence = float(probs[prediction])

    label_map = label_map or {}
    label = label_map.get(prediction, f"Class {prediction}")

    return {
        "prediction": prediction,
        "label": label,
        "confidence": confidence,
        "probabilities": probs,
        "model": model,
        "X_raw": tabular_raw,
        "X_scaled": X_scaled,
        "feature_columns": tabular_columns,
        "embedding_columns": embedding_columns,
    }
