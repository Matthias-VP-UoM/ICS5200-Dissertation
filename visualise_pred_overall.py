from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import json
import shap

from utils.live_features import extract_live_features
from utils.live_prediction import LABEL_MAP, predict_accessibility

import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Setup Directories
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "visualisations_shap_overall"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = PROJECT_ROOT / "outputs" / "models_final"

CSV_PATH = PROJECT_ROOT / "data" / "test_split_v2.csv"
test_records = pd.read_csv(CSV_PATH)

MODEL_CONFIG = {
    "Random Forest": 'rf_model_v2.pkl',
    "Gradient Boosting": 'gb_model_v2.pkl',
    "ResNet18": 'resnet18_model_v2.pth',
    "ResNet50": 'resnet50_model_v2.pth',
    "VGG16": 'vgg16_model_v2.pth',
    "ViT-B-16": 'vit_b_16_model_v2.pth',
    "Swin-T": 'swin_t_model_v2.pth',
}

def convert_pred_to_str(pred_result):
    if pred_result == 0:
        return "High"
    elif pred_result == 1:
        return "Medium"
    elif pred_result == 2:
        return "Low"
    else:
        raise ValueError("Number not in range")

def generate_global_shap_plots(test_df: pd.DataFrame, tabular_model_names: list[str]):
    print("\n==========================================")
    print("EXTRACTING FEATURES FOR GLOBAL SHAP PLOTS")
    print("==========================================")
    
    # Collect all tabular samples cleanly into a single 2D list
    features_list = []
    for index, row in test_df.iterrows():
        screenshot_path = str(row["screenshot_path"])
        html_path = str(row["html_path"])
        try:
            feats = extract_live_features(screenshot_path, html_path)
            
            # If extract_live_features returns a DataFrame or Series, flatten to dict/row
            if isinstance(feats, pd.DataFrame):
                feats = feats.iloc[0].to_dict()
            elif isinstance(feats, pd.Series):
                feats = feats.to_dict()

            features_list.append(feats)
        except Exception as e:
            print(f"Failed feature extraction for row {index}: {e}")

    if not features_list:
        print("No valid feature samples extracted.")
        return

    # Convert list of dicts directly into a clean 2D DataFrame (N_samples, N_features)
    X_raw = pd.DataFrame(features_list)

    # Compute and Plot Global SHAP for each specified model
    for model_name in tabular_model_names:
        print(f"\nComputing Global SHAP summary for: {model_name}...")
        
        # Pass a single record dict (1D mapping) to predict_accessibility
        first_sample_dict = X_raw.iloc[0].to_dict()
        sample_res = predict_accessibility(first_sample_dict, model_name)
        model = sample_res["model"]
        
        # Scale/preprocess full test dataset using model scaler if present
        if "scaler" in sample_res and sample_res["scaler"] is not None:
            X_scaled = sample_res["scaler"].transform(X_raw)
        elif "X_scaled" in sample_res:
            # If predict_accessibility handles scaling internally, scale full DataFrame
            scaler = getattr(model, "scaler", None)
            X_scaled = scaler.transform(X_raw) if scaler else X_raw.values
        else:
            X_scaled = X_raw.values

        # Extract underlying classifier if encapsulated within an imblearn/sklearn Pipeline
        classifier = model.named_steps["classifier"] if hasattr(model, "named_steps") else model
        
        # Calculate SHAP Values
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_scaled)

        # Plot Global Summary
        plt.figure(figsize=(10, 6))
        
        # Plot summary across features
        shap.summary_plot(shap_values, X_scaled, feature_names=X_raw.columns, show=False)

        # Save plot
        output_path = OUTPUT_DIR / f"{model_name.lower().replace(' ', '_')}_global_shap_summary.png"
        plt.title(f"Global SHAP Summary - {model_name}", fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  └─ Global SHAP figure saved to: {output_path}")


if __name__ == "__main__":
    # Run global SHAP visualisations for both tabular models across all test samples
    tabular_models = ["Random Forest", "Gradient Boosting"]
    generate_global_shap_plots(test_df=test_records, tabular_model_names=tabular_models)