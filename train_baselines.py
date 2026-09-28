import os
import pandas as pd
import pickle
import time
import psutil

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler

from utils.data_loader import load_dataset, split_and_scale_data
from models.random_forests import train_random_forest
from models.gradient_boosting import train_grad_boost
# from utils.shap_utils import generate_global_shap_plots

# Utility function to print out memory (RAM) and processing (CPU) usage
def log_resource_usage(stage_name=""):
    process = psutil.Process(os.getpid())

    # RAM usage in Megabytes
    ram_usage_mb = process.memory_info().rss / (1024 * 1024)

    # CPU usage percentage (interval calculates usage over 0.1 seconds)
    cpu_usage_pct = psutil.cpu_percent(interval=1)

    print(f"--- [Resource Log: {stage_name}] ---")
    print(f"RAM Usage: {ram_usage_mb:.2f} MB")
    print(f"System CPU Usage: {cpu_usage_pct}%")
    print("-" * 35)


def main():
    # Initialising output paths and directories
    output_dir = 'outputs'
    models_dir = 'models'
    results_dir = 'results'
    plots_dir = 'plots'
    fi_dir = 'feature_importance'

    models_path = os.path.join(output_dir, models_dir)
    results_path = os.path.join(output_dir, results_dir)
    plots_path = os.path.join(output_dir, plots_dir)
    fi_path = os.path.join(output_dir, fi_dir)

    for p in [models_path, results_path, plots_path, fi_path]:
        if not os.path.exists(p):
            os.makedirs(p)

    # Loading the full dataset
    data_dir = 'data'

    DATASET_PATH = os.path.join(data_dir, 'annotated_dataset.csv')

    TRAIN_SPLIT_PATH = os.path.join(data_dir, 'train_split.csv')
    TEST_SPLIT_PATH = os.path.join(data_dir, 'test_split.csv')

    df = pd.read_csv(DATASET_PATH)

    df['screenshot_path'] = df['screenshot_path'].str.replace('\\', '/', regex=False)

    print("Dataset loaded successfully.")
    print("Dataset shape:", df.shape)

    # Checks if a train and test set already exist
    if os.path.exists(TRAIN_SPLIT_PATH) and os.path.exists(TEST_SPLIT_PATH):
        print("Found existing train/test split files! Loading them to ensure identical subsets...")
        train_df = pd.read_csv(TRAIN_SPLIT_PATH)
        test_df = pd.read_csv(TEST_SPLIT_PATH)
        
        train_df['screenshot_path'] = train_df['screenshot_path'].str.replace('\\', '/', regex=False)
        test_df['screenshot_path'] = test_df['screenshot_path'].str.replace('\\', '/', regex=False)
    else:
        print(" No existing split files found. Generating new master split and saving to disk...")
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df["accessibility_label"]
        )
        train_df.to_csv(TRAIN_SPLIT_PATH, index=False)
        test_df.to_csv(TEST_SPLIT_PATH, index=False)

    # Initialise the classical machine learning pipeline

    print("\n==============================")
    print("CLASSICAL ML PIPELINE")
    print("==============================")

    X, y, feature_columns = load_dataset(DATASET_PATH)

    train_ids = train_df["id"].astype(str).str.zfill(3)
    test_ids = test_df["id"].astype(str).str.zfill(3)

    df["id"] = df["id"].astype(str).str.zfill(3)

    train_mask = df["id"].isin(train_ids)
    test_mask = df["id"].isin(test_ids)

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]
    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    # Oversample the non-majority class to provide more instances for the model to train
    # NOTE: This is only done if the USE_RANDOM_OVERSAMPLING variable is set to True
    USE_RANDOM_OVERSAMPLING = True

    print("\nOriginal class distribution:")
    print(y_train.value_counts().sort_index())

    if USE_RANDOM_OVERSAMPLING:
        oversampler = RandomOverSampler(
            sampling_strategy="not majority",
            random_state=42
        )

        X_train_resampled, y_train_resampled = oversampler.fit_resample(
            X_train,
            y_train
        )

        print("\nClass distribution after oversampling:")
        print(y_train_resampled.value_counts().sort_index())
    else:
        X_train_resampled = X_train.copy()
        y_train_resampled = y_train.copy()

    # Normalise/Scale features safely across the isolated sets
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    # Optimise and train Random Forest model
    rf_model, rf_results = train_random_forest(
        X_train_scaled,
        X_test_scaled,
        y_train_resampled,
        y_test,
        feature_columns,
        use_optuna_tuning=False
    )

    # Save the Random Forest model for future use
    rf_model_path = os.path.join(models_path, 'rf_model.pkl')

    with open(rf_model_path, 'wb') as f:
        pickle.dump(rf_model, f)

    # Optimise and train Gradient Boosting model
    gb_model, gb_results = train_grad_boost(
        X_train_scaled,
        X_test_scaled,
        y_train_resampled,
        y_test,
        feature_columns,
        use_optuna_tuning=False
    )

    # Save the Gradient Boosting model for future use
    gb_model_path = os.path.join(models_path, 'gb_model.pkl')

    with open(gb_model_path, 'wb') as f:
        pickle.dump(gb_model, f)

    # Save the utilised scaler for future use
    scaler_path = os.path.join(models_path, 'scaler.pkl')

    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Generate sample-level predictions
    rf_preds = rf_model.predict(X_test_scaled)
    gb_preds = gb_model.predict(X_test_scaled)

    # Construct table linking original IDs and ground truth labels
    predictions_df = pd.DataFrame({
        "id": test_df["id"].values,
        "actual_label": y_test.values,
        "rf_predicted_label": rf_preds,
        "gb_predicted_label": gb_preds
    })

    # Adding class prediction probabilities
    rf_probs = rf_model.predict_proba(X_test_scaled)
    gb_probs = gb_model.predict_proba(X_test_scaled)

    for class_idx in range(rf_probs.shape[1]):
        predictions_df[f"rf_prob_class_{class_idx}"] = rf_probs[:, class_idx]
        predictions_df[f"gb_prob_class_{class_idx}"] = gb_probs[:, class_idx]

    # Save to CSV in results directory
    predictions_output_path = os.path.join(results_path, "individual_predictions.csv")
    predictions_df.to_csv(predictions_output_path, index=False)

    print(f"\nIndividual predictions saved successfully to: {predictions_output_path}")

    # Saving the comparison results

    comparison_df = pd.DataFrame([
        rf_results,
        gb_results
    ])

    comparison_df.to_csv(
        "outputs/results/model_comparison.csv",
        index=False
    )

    print("\nAll models trained and SHAP figures generated successfully.")


if __name__ == "__main__":
    main()