from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"

# Defines a mapping for converting model predictions from integer to string
LABEL_MAP = {
    0: "High Accessibility",
    1: "Medium Accessibility",
    2: "Low Accessibility",
}

# FEATURE_COLUMNS = [
#     "num_links",
#     "num_images",
#     "num_buttons",
#     "num_paragraphs",
#     "num_headings",
#     "word_count",
#     "dom_depth",
#     "num_advertisement_iframes",
#     "num_non_ad_iframes",
#     "edge_density",
#     "contrast",
#     "colour_variance",
#     "layout_density",
#     "text_density_proxy",
#     "brightness_variance",
# ]

# Define features that to be used to compute the prediction for a given input
FEATURE_COLUMNS = [
    # Screenshot-based visual features
    "edge_density",
    "contrast",
    "colour_variance",
    # "layout_density",
    "text_density_proxy",
    "brightness_variance",
    "whitespace_ratio",

    # DOM and content features
    "dom_depth",
    "num_links",
    "num_images",
    "num_buttons",
    "num_forms",
    "num_paragraphs",
    "num_headings",
    "word_count",

    # Advertisement and embedded-content features
    "num_advertisement_iframes",
    "num_non_ad_iframes",

    # ARIA and accessible-naming features
    "num_aria_attributes",
    "aria_attribute_density",
    "num_aria_label",
    "num_aria_labelledby",
    "num_aria_hidden",
    "num_aria_live",
    # "interactive_elements_without_label",

    # Image accessibility features
    "num_images_without_alt",
    "alt_text_coverage",

    # Heading accessibility features
    "heading_level_skips",
    "empty_heading_count",

    # Semantic landmark features
    "has_main_landmark",
    "has_nav_landmark",
    "semantic_element_ratio",

    # Accessible-name and form features
    # "buttons_without_accessible_name",
    # "links_without_accessible_name",
    # "inputs_without_label",
    "unlabelled_button_ratio",
    "unlabelled_link_ratio",
    "unlabelled_input_ratio",

    # Readability feature
    "average_paragraph_word_count",
]

# Loads the required classical models for prediction
@lru_cache(maxsize=1)
def load_models():
    rf_model = joblib.load(MODEL_DIR / "rf_model.pkl")
    gb_model = joblib.load(MODEL_DIR / "gb_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    return rf_model, gb_model, scaler


def prepare_feature_frame(features: dict) -> pd.DataFrame:
    df = pd.DataFrame([features])
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    for column in missing:
        df[column] = 0.0

    X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X

# Predicts the level of accessibility for a given webpage screenshot and returns
# a series of metrics in dictionary form
def predict_accessibility(features: dict, model_choice: str = "Random Forest"):
    rf_model, gb_model, scaler = load_models()
    model = rf_model if model_choice == "Random Forest" else gb_model

    X_raw = prepare_feature_frame(features)
    X_scaled_array = scaler.transform(X_raw)
    X_scaled = pd.DataFrame(X_scaled_array, columns=FEATURE_COLUMNS)

    # prediction = int(model.predict(X_scaled_array)[0])
    # probabilities = np.asarray(model.predict_proba(X_scaled_array)[0], dtype=float)

    # Model inference
    preds = model.predict(X_scaled_array)
    probs = model.predict_proba(X_scaled_array)[0]
    
    prediction = int(preds[0])
    confidence = float(probs[prediction])

    # Map the numerical class prediction to your text label map
    label = LABEL_MAP.get(prediction, f"Class {prediction}")

    return {
        "prediction": prediction,
        "label": label,
        "confidence": confidence,
        "probabilities": probs,
        "model": model,
        "X_raw": X_raw,
        "X_scaled": X_scaled,
        "feature_columns": FEATURE_COLUMNS,
    }
