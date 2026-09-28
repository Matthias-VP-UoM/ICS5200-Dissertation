from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import json

from utils.explainability import explain_tabular_prediction, make_shap_bar_figure, generate_per_class_beeswarm
from utils.gradcam import create_gradcam_overlay
from utils.live_features import extract_live_features
from utils.live_prediction import LABEL_MAP, predict_accessibility

import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Setup Directories
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "visualisations_final_5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = PROJECT_ROOT / "outputs" / "models_final"

CSV_PATH = PROJECT_ROOT / "data" / "test_split_v6.csv"
test_records = pd.read_csv(CSV_PATH)

MODEL_CONFIG = {
    "Random Forest": 'rf_model_v7.pkl',
    "Gradient Boosting": 'gb_model_v7.pkl',
    "ResNet18": 'resnet18_model_v3.pth',
    "ResNet50": 'resnet50_model_v3.pth',
    "VGG16": 'vgg16_model_v3.pth',
    "ViT-B-16": 'vit_b_16_model_v3.pth',
    "Swin-T": 'swin_t_model_v3.pth',
}

# Function used to convert the prediction result (returned as an integer) into a string
# This is done according to how the predictions are assigned in the dataset
def convert_pred_to_str(pred_result):
    if pred_result == 0:
        return "High"
    elif pred_result == 1:
        return "Medium"
    elif pred_result == 2:
        return "Low"
    else:
        raise ValueError("Number not in range")

# Function which goes through the test set and generates the SHAP and GRAD-CAM
# visualisations for each prediction
def process_test_dataset(
    test_df: pd.DataFrame,
    tabular_model_name: str = "Random Forest",
    visual_architecture: str = "ResNet18",
    cam_method: str = "layercam",  # Options: "layercam", "gradcam++", "guidedgradcam", "gradcam"
    checkpoint_path: Path | str | None = None,
):
    tabular_saved = False
    visual_saved = True

    if checkpoint_path is None:
        checkpoint_path = MODEL_DIR / f"{MODEL_CONFIG.get(visual_architecture)}"

    str_checkpoint_path = str(checkpoint_path)

    for index, row in test_df.iterrows():
        record_id = row.get("id", f"sample_{index}")
        screenshot_path = str(row["screenshot_path"])
        html_path = str(row["html_path"])

        print(f"[{index + 1}/{len(test_df)}] Processing record: {record_id}...")

        try:
            # Extract features and generate prediction
            features = extract_live_features(screenshot_path, html_path)
            result = predict_accessibility(features, tabular_model_name)
            result_str = convert_pred_to_str(result["prediction"])
            result_conf = f"{result['confidence']:.2%}"

            target_dir = OUTPUT_DIR / tabular_model_name.lower()
            target_dir.mkdir(parents=True, exist_ok=True)
            shap_output_path = target_dir / f"{record_id}_shap.png"
            beeswarm_shap_output_path = target_dir / f"{record_id}_beeswarm_shap.png"

            if not tabular_saved:
                shap_df = explain_tabular_prediction(
                    result["model"], result["X_scaled"], result["prediction"]
                )
                shap_fig = make_shap_bar_figure(shap_df, result_str, result_conf)
                shap_fig.savefig(shap_output_path, bbox_inches="tight", dpi=150)
                plt.close(shap_fig)
                beeswarm_fig = generate_per_class_beeswarm(
                    model=result["model"],
                    X_test=result["X_scaled"],
                    class_names=["High", "Medium", "Low"],
                )
                beeswarm_fig.savefig(beeswarm_shap_output_path, bbox_inches="tight", dpi=150)
                plt.close(beeswarm_fig)

            # Generate a GRAD-CAM overlay within the processed image
            gradcam_overlay, visual_probs, visual_class, cam_metrics = create_gradcam_overlay(
                screenshot_path=screenshot_path,
                checkpoint_path=str_checkpoint_path,
                architecture=visual_architecture,
                method=cam_method,
            )

            visual_class_str = convert_pred_to_str(visual_class)
            visual_prob_max = visual_probs[visual_class]
            visual_best_prob = f"{visual_prob_max:.2%}"

            visual_architecture_dir = f"{visual_architecture.replace('/', '-').lower()}_{cam_method.lower()}"
            gradcam_target_dir = OUTPUT_DIR / visual_architecture_dir
            gradcam_target_dir.mkdir(parents=True, exist_ok=True)

            visual_metrics_dir = OUTPUT_DIR / visual_architecture_dir / f"{cam_method.lower()}_metrics"
            visual_metrics_dir.mkdir(parents=True, exist_ok=True)
            # visual_metrics_dir.makedirs(parents=True, exist_ok=True)

            gradcam_output_path = gradcam_target_dir / f"{record_id}_{cam_method.lower()}.png"
            gradcam_output_metrics_path = visual_metrics_dir / f"{record_id}_{cam_method.lower()}_metrics.json"

            if not visual_saved:
                gradcam_overlay.save(gradcam_output_path)
            
            with open(gradcam_output_metrics_path, "w+") as f:
                json.dump(cam_metrics, f, indent=4)

            print(
                f"  └─ Saved SHAP: {shap_output_path.name} | Pred: {result_str} ({result_conf})\n"
                f"  └─ Saved {cam_method.upper()}: {gradcam_output_path.name} | Pred: {visual_class_str} ({visual_best_prob})\n"
                f"  └─ Metrics -> Drop-in-Confidence: {cam_metrics['drop_in_confidence']:.2%}, Masked Conf: {cam_metrics['masked_confidence']:.2%}"
            )

        except Exception as exc:
            print(f"  └─ [ERROR] Failed to process {record_id}: {exc}")


if __name__ == "__main__":
    subset_df = test_records.iloc[152:]
    for tab_model in ["Random Forest", "Gradient Boosting"]:
        for vis_model in ["ResNet18", "ResNet50", "VGG16", "ViT-B-16", "Swin-T"]:
            for cam_method in ["layercam", "gradcam++", "guidedgradcam"]:
                print(f"Running {tab_model}+{vis_model} with {cam_method} method...")
                process_test_dataset(
                    test_df=test_records,
                    tabular_model_name=tab_model,
                    visual_architecture=vis_model,  
                    cam_method=cam_method
                )