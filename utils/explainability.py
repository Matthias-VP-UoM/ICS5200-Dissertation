from __future__ import annotations

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st

def select_class_values(shap_values, predicted_class: int, n_features: int) -> np.ndarray:
    if isinstance(shap_values, list):
        values = np.asarray(shap_values[predicted_class])
        return values[0]

    values = np.asarray(shap_values)

    if values.ndim == 3:
        if values.shape[1] == n_features:
            return values[0, :, predicted_class]
        if values.shape[2] == n_features:
            return values[0, predicted_class, :]

    if values.ndim == 2:
        return values[0]

    raise ValueError(f"Unsupported SHAP value shape: {values.shape}")

def unwrap_classifier(model):
    if hasattr(model, "named_steps"):
        if "classifier" in model.named_steps:
            return model.named_steps["classifier"]
        # Fall back to the final step of the pipeline
        return model.steps[-1][1]
    return model

# Generates the SHAP values for the classical model prediction
def explain_tabular_prediction(model, X_scaled, predicted_class):
    try:
        base_estimator = unwrap_classifier(model)
        if "GradientBoosting" in type(base_estimator).__name__:
            # Convert DataFrame to a raw NumPy array to match how the model was fitted
            # X_raw_array = X_scaled.values
            background_data = np.zeros((2, X_scaled.shape[1]))
            
            # Pass the raw array and the model's prediction function
            explainer = shap.PermutationExplainer(model.predict_proba, background_data)
            
            # Calculate SHAP values using the raw array configuration
            shap_values = explainer(X_scaled)
            
            # Extract the correct class slice from the resulting list or array matrix
            if hasattr(shap_values, "values") and len(shap_values.values.shape) == 3:
                actual_shap = shap_values.values[0, :, predicted_class]
            else:
                actual_shap = shap_values[0]
                
        else:
            # Standard path for Random Forest models (which accept DataFrames fine)
            explainer = shap.TreeExplainer(base_estimator)
            shap_values = explainer.shap_values(X_scaled, check_additivity=False)

            if hasattr(shap_values, "values"):
                vals = shap_values.values
                if vals.ndim == 3:
                    # Shape: (samples, features, classes)
                    actual_shap = vals[0, :, predicted_class]
                else:
                    actual_shap = vals[0]

            elif isinstance(shap_values, list):
                actual_shap = shap_values[predicted_class][0]

            # Case C: shap_values is a 3D numpy array
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                if shap_values.shape[1] == X_scaled.shape[1]:
                    # Shape: (samples, features, classes)
                    actual_shap = shap_values[0, :, predicted_class]
                else:
                    # Shape: (samples, classes, features)
                    actual_shap = shap_values[0, predicted_class, :]
            else:
                actual_shap = shap_values[0]

        # Flatten the array safely into 1-dimension
        actual_shap = np.asarray(actual_shap).ravel()

        # Build the final DataFrame mapping the calculated values back to your features
        shap_df = pd.DataFrame({
            "Feature": X_scaled.columns,
            "SHAP value": actual_shap,
            "Absolute impact": np.abs(actual_shap)
        }).sort_values(by="Absolute impact", ascending=False)
        
        return shap_df

    except Exception as e:
        st.exception(e)
        # st.warning(f"Could not generate SHAP visualization: {e}")
        # return pd.DataFrame({
        #     "Feature": X_scaled.columns,
        #     "SHAP value": 0.0,
        #     "Absolute impact": 0.0
        # })

# Plots the SHAP bar chart for the generated prediction
def make_shap_bar_figure(explanation_df: pd.DataFrame, pred_result, pred_class_prob, top_n: int = 12):
    plot_df = explanation_df.head(top_n).sort_values("SHAP value")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(plot_df["Feature"], plot_df["SHAP value"])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("SHAP contribution to the predicted class")
    ax.set_ylabel("Feature")
    ax.set_title(f"SHAP Explainability - Prediction: {pred_result}, Confidence: {pred_class_prob}")
    fig.tight_layout()
    return fig

# Generates per-class SHAP beeswarm plots using minimal input parameters
# Infers feature names directly from X_test and model details automatically
def generate_per_class_beeswarm(model, X_test: pd.DataFrame, shap_values=None, class_names: list[str] | None = None, top_n: int = 10) -> plt.Figure | None:
    base_estimator = unwrap_classifier(model)
    feature_columns = list(X_test.columns)
    model_name = getattr(base_estimator, "__class__", type(base_estimator)).__name__

    # Compute SHAP values dynamically if not pre-calculated
    if shap_values is None:
        if "GradientBoosting" in type(base_estimator).__name__: # Gradient Boosting - KernelExplainer
            X_background = shap.kmeans(X_test, min(50, len(X_test))).data
            explainer = shap.KernelExplainer(base_estimator.predict_proba, X_background)
            raw_shap_values = explainer.shap_values(
                X_test, nsamples=2 * len(feature_columns) + 1
            )

            vals = (
                np.stack(raw_shap_values, axis=-1)
                if isinstance(raw_shap_values, list)
                else raw_shap_values
            )

            expected_value = explainer.expected_value
            if expected_value is None:
                expected_value = base_estimator.predict_proba(X_background).mean(axis=0)

            base_vals = (
                np.array(expected_value)
                if isinstance(expected_value, list)
                else np.asarray(expected_value)
            )
            n_samples, n_classes = vals.shape[0], vals.shape[-1]
            if base_vals.ndim == 1 and base_vals.shape[0] == n_classes:
                base_vals = np.tile(base_vals, (n_samples, 1))

            shap_values = shap.Explanation(
                values=vals,
                base_values=base_vals,
                data=X_test.values,
                feature_names=feature_columns,
            )
        else:   # Random Forest - TreeExplainer
            explainer = shap.TreeExplainer(base_estimator)
            shap_values = explainer(X_test, check_additivity=False)

    # Extract values and validate multiclass dimensions
    vals = shap_values.values if hasattr(shap_values, "values") else shap_values

    if vals.ndim != 3:
        print("Model output is not multiclass (values.ndim != 3); skipping per-class breakdown.")
        return None

    n_samples, n_features, n_classes = vals.shape

    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    # Identify Top-N Features by overall mean absolute SHAP
    mean_abs_shap = np.abs(vals).mean(axis=(0, 2))
    top_idx = np.argsort(mean_abs_shap)[::-1][:top_n]
    top_features = [feature_columns[i] for i in top_idx]

    X_top = X_test[top_features]

    # Build Subplots per Class
    fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 6), sharex=False)
    if n_classes == 1:
        axes = [axes]

    for c in range(n_classes):
        class_vals = vals[:, top_idx, c]

        class_base = getattr(shap_values, "base_values", None)
        if class_base is not None:
            if hasattr(class_base, "ndim") and class_base.ndim == 2:
                class_base = class_base[:, c]
            elif hasattr(class_base, "ndim") and class_base.ndim == 1 and class_base.shape[0] == n_classes:
                class_base = np.full(n_samples, class_base[c])
        else:
            class_base = np.zeros(n_samples)

        class_exp = shap.Explanation(
            values=class_vals,
            base_values=class_base,
            data=X_top.values,
            feature_names=top_features,
        )

        plt.sca(axes[c])
        shap.summary_plot(class_exp, X_top, show=False, plot_size=None)
        axes[c].set_title(f"{model_name} — {class_names[c]}", fontsize=12)

    plt.tight_layout()

    return fig