import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(model_name, y_test, predictions):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
    recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
    f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

    print(f"\n{model_name} Results")
    print("-" * 30)
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1-Score :", f1)

    print("\nClassification Report")
    print(classification_report(y_test, predictions))

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def save_feature_importance(model, feature_columns, output_path=None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    feature_importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_
    })

    feature_importance_df = feature_importance_df.sort_values(
        by="importance",
        ascending=False
    )

    if output_path:
        if not os.path.exists(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        feature_importance_df.to_csv(output_path, index=False)

    print("\nTop 10 Features")
    print(feature_importance_df.head(10))

    return feature_importance_df


def save_classification_report(y_test, predictions, output_path=None):
    report_dict = classification_report(y_test, predictions, output_dict=True)

    report_df = pd.DataFrame(report_dict).transpose()

    if output_path:
        if not os.path.exists(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            report_df.to_csv(output_path, index=True)

    print("\nClassification Report Saved successfully!")
    print(report_df)


def plot_confusion_matrix(model_name, y_test, predictions, output_path=None):
    matrix = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")

    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    if output_path:
        if not os.path.exists(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)

    plt.show()

# Generates global SHAP summary/beeswarm plots across the test dataset for a given model
def generate_global_shap_plots(model, X_test, feature_columns, model_name, output_dir="outputs/plots_shap"):
    os.makedirs(output_dir, exist_ok=True)

    # Unwrap pipeline classifier if present
    if hasattr(model, "named_steps") and "classifier" in model.named_steps:
        base_estimator = model.named_steps["classifier"]
    else:
        base_estimator = model

    X_test_df = pd.DataFrame(X_test, columns=feature_columns)

    if "GradientBoosting" in type(base_estimator).__name__:
        X_background = shap.kmeans(X_test_df, 50).data
        explainer = shap.KernelExplainer(base_estimator.predict_proba, X_background)

        raw_shap_values = explainer.shap_values(
            X_test_df,
            nsamples=2 * len(feature_columns) + 1
        )

        if isinstance(raw_shap_values, list):
            vals = np.stack(raw_shap_values, axis=-1)
        else:
            vals = raw_shap_values

        expected_value = explainer.expected_value
        if expected_value is None:
            expected_value = base_estimator.predict_proba(X_background).mean(axis=0)

        base_vals = np.array(expected_value) if isinstance(expected_value, list) else np.asarray(expected_value)

        n_samples = vals.shape[0]
        n_classes = vals.shape[-1]
        if base_vals.ndim == 1 and base_vals.shape[0] == n_classes:
            base_vals = np.tile(base_vals, (n_samples, 1))

        shap_values = shap.Explanation(
            values=vals,
            base_values=base_vals,
            data=X_test_df.values,
            feature_names=list(feature_columns)
        )
    else:
        # Fast path for Random Forest and standard models
        explainer = shap.TreeExplainer(base_estimator)
        shap_values = explainer(X_test_df, check_additivity=False)
    # explainer = shap.TreeExplainer(base_estimator)
    # shap_values = explainer(X_test_df, check_additivity=False)

    # Extract raw numpy values from the Explanation object
    vals = shap_values.values if hasattr(shap_values, "values") else shap_values

    # Average mean absolute impact across samples (and classes if multiclass)
    if vals.ndim == 3:
        # Shape: (samples, features, classes) -> average across samples (axis 0) and classes (axis 2)
        mean_abs_shap = np.abs(vals).mean(axis=(0, 2))
    else:
        # Shape: (samples, features) -> average across samples (axis 0)
        mean_abs_shap = np.abs(vals).mean(axis=0)

    shap_df = pd.DataFrame(
        mean_abs_shap,
        index=feature_columns,
        columns=["Mean |SHAP|"]
    ).sort_values("Mean |SHAP|", ascending=False)

    print("Top Global Features:")
    print(shap_df.head(15))

    # Plot & Save Global Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_df, feature_names=feature_columns, show=False)
    ax = plt.gca()

    row_totals = {}
    for p in ax.patches:
        width = p.get_width()
        if width > 0.015:  # Only annotate segments wide enough to be readable
            x_center = p.get_x() + width / 2
            y_center = p.get_y() + p.get_height() / 2

            # Add subtotal inside the specific class sub-bar segment
            ax.text(
                x_center,
                y_center,
                f"{width:.3f}",
                va='center',
                ha='center',
                fontsize=7.5,
                color='white',
                weight='bold'
            )

        # Track total bar length per feature row
        y = p.get_y()
        row_totals[y] = row_totals.get(y, 0) + width

    # Add the overall sum at the end of each stacked bar
    for y, total_val in row_totals.items():
        if total_val > 0:
            ax.text(
                total_val + 0.002,
                y + 0.25,
                f"Total: {total_val:.3f}",
                va='center',
                ha='left',
                fontsize=8.5,
                color='black',
                weight='bold'
            )

    # Extend x-axis limit slightly so labels don't get clipped
    max_x = max(row_totals.values()) if row_totals else 0.15
    ax.set_xlim(0, max_x * 1.15)

    legend = ax.get_legend()
    if legend is not None:
        for text_obj, new_label in zip(legend.get_texts(), ["High", "Medium", "Low"]):
            text_obj.set_text(new_label)

    plt.title(f"{model_name} Global SHAP Feature Importance", fontsize=14, pad=15)
    plt.xlabel("SHAP value (Impact on Model Output)")
    plt.tight_layout()

    file_name = f"{model_name.lower().replace(' ', '_')}_shap_summary.png"
    save_path = os.path.join(output_dir, file_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"SHAP summary plot successfully saved to: {save_path}")

    return shap_values

# Generates per-class SHAP beeswarm plots for the top-N globally important features
def generate_per_class_beeswarm(shap_values, X_test, feature_columns, model_name, class_names=None, top_n=10, output_dir="outputs/plots_shap"):
    os.makedirs(output_dir, exist_ok=True)

    X_test_df = pd.DataFrame(X_test, columns=feature_columns)

    vals = shap_values.values if hasattr(shap_values, "values") else shap_values

    if vals.ndim != 3:
        print("Model output is not multiclass (values.ndim != 3) — a standard "
              "shap.summary_plot beeswarm already shows this; skipping per-class breakdown.")
        return

    n_samples, n_features, n_classes = vals.shape

    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    # Rank features by overall mean |SHAP| (same ranking as your stacked bar)
    mean_abs_shap = np.abs(vals).mean(axis=(0, 2))
    top_idx = np.argsort(mean_abs_shap)[::-1][:top_n]
    top_features = [feature_columns[i] for i in top_idx]

    X_top = X_test_df[top_features]

    fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 6), sharex=False)
    if n_classes == 1:
        axes = [axes]

    for c in range(n_classes):
        # Build a 2D Explanation for this class, restricted to the top features
        class_vals = vals[:, top_idx, c]

        class_base = shap_values.base_values
        if hasattr(class_base, "ndim") and class_base.ndim == 2:
            class_base = class_base[:, c]
        elif hasattr(class_base, "ndim") and class_base.ndim == 1 and class_base.shape[0] == n_classes:
            class_base = np.full(n_samples, class_base[c])

        class_exp = shap.Explanation(
            values=class_vals,
            base_values=class_base,
            data=X_top.values,
            feature_names=top_features
        )

        plt.sca(axes[c])
        shap.summary_plot(class_exp, X_top, show=False, plot_size=None)
        axes[c].set_title(f"{model_name} — {class_names[c]} Accessibility", fontsize=12)

    plt.tight_layout()
    file_name = f"{model_name.lower().replace(' ', '_')}_shap_beeswarm_per_class.png"
    save_path = os.path.join(output_dir, file_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Per-class beeswarm plot saved to: {save_path}")
    return top_features

# Generates global SHAP summary/beeswarm plots across the test dataset for a given model
def generate_shap_error_plots(model, X_test, y_test, y_pred, feature_columns, model_name, shap_values, output_dir="outputs/plots_shap"):
    os.makedirs(output_dir, exist_ok=True)

    X_test_df = pd.DataFrame(X_test, columns=feature_columns)

    misclassified_idx = np.where(y_pred != y_test)[0]

    # Inspect SHAP values strictly for misclassified instances to diagnose failure modes
    misclassified_shap = shap_values[misclassified_idx]

    # Ensure misclassified_shap is extracted as a NumPy array
    raw_misclassified = misclassified_shap.values if hasattr(misclassified_shap, "values") else misclassified_shap

    # Handle shape based on binary (2D) vs multiclass (3D) SHAP output
    if raw_misclassified.ndim == 3:
        # Shape: (samples, features, classes) -> average across samples (axis 0) & classes (axis 2)
        mean_error_shap = np.abs(raw_misclassified).mean(axis=(0, 2))
    else:
        # Shape: (samples, features) -> average across samples (axis 0)
        mean_error_shap = np.abs(raw_misclassified).mean(axis=0)

    misclassified_shap_df = pd.DataFrame({
        'Feature': X_test_df.columns,
        'Mean_Abs_SHAP_Error': mean_error_shap
    }).sort_values(by='Mean_Abs_SHAP_Error', ascending=False)

    print("Top Global Features:")
    print(misclassified_shap_df.head(15))

    # Plot & Save Global Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_df, feature_names=feature_columns, show=False)
    ax = plt.gca()

    row_totals = {}
    for p in ax.patches:
        width = p.get_width()
        if width > 0.015:  # Only annotate segments wide enough to be readable
            x_center = p.get_x() + width / 2
            y_center = p.get_y() + p.get_height() / 2

            # Add subtotal inside the specific class sub-bar segment
            ax.text(
                x_center,
                y_center,
                f"{width:.3f}",
                va='center',
                ha='center',
                fontsize=7.5,
                color='white',
                weight='bold'
            )

        # Track total bar length per feature row
        y = p.get_y()
        row_totals[y] = row_totals.get(y, 0) + width

    # Add the overall sum at the end of each stacked bar
    for y, total_val in row_totals.items():
        if total_val > 0:
            ax.text(
                total_val + 0.002,
                y + 0.25,
                f"Total: {total_val:.3f}",
                va='center',
                ha='left',
                fontsize=8.5,
                color='black',
                weight='bold'
            )

    # Extend x-axis limit slightly so labels don't get clipped
    max_x = max(row_totals.values()) if row_totals else 0.15
    ax.set_xlim(0, max_x * 1.15)

    legend = ax.get_legend()
    if legend is not None:
        for text_obj, new_label in zip(legend.get_texts(), ["High", "Medium", "Low"]):
            text_obj.set_text(new_label)

    plt.title(f"{model_name} Global SHAP Feature Importance", fontsize=14, pad=15)
    # plt.xlabel("SHAP value (Impact on Model Output)")
    plt.tight_layout()

    file_name = f"{model_name.lower().replace(' ', '_')}_shap_error_summary.png"
    save_path = os.path.join(output_dir, file_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"SHAP error summary plot successfully saved to: {save_path}")

# Compares global mean absolute SHAP values against misclassification SHAP values
def compare_global_vs_error_shap(shap_values, X_test, y_test, y_pred, feature_columns, output_path=None):
    raw_shap = shap_values.values if hasattr(shap_values, "values") else shap_values
    mis_idx = np.where(y_pred != y_test)[0]

    # Global Mean Absolute Impact
    global_impact = np.abs(raw_shap).mean(axis=(0, 2)) if raw_shap.ndim == 3 else np.abs(raw_shap).mean(axis=0)

    # Error Mean Absolute Impact
    raw_mis = raw_shap[mis_idx]
    error_impact = np.abs(raw_mis).mean(axis=(0, 2)) if raw_mis.ndim == 3 else np.abs(raw_mis).mean(axis=0)

    if feature_columns is not None:
        features = feature_columns
    elif hasattr(X_test, "columns"):
        features = X_test.columns
    else:
        features = [f"F_{i}" for i in range(raw_shap.shape[1])]

    df = pd.DataFrame({
        'Feature': features,
        'Global_Impact': global_impact,
        'Error_Impact': error_impact
    })

    # Calculate Ranks and Shifts
    df['Global_Rank'] = df['Global_Impact'].rank(ascending=False).astype(int)
    df['Error_Rank'] = df['Error_Impact'].rank(ascending=False).astype(int)
    df['Rank_Shift'] = df['Global_Rank'] - df['Error_Rank'] # Positive = higher priority in errors

    final_df = df.sort_values(by='Error_Rank')

    # with open(output_path, "w") as f:
    final_df.to_csv(output_path, index=True)