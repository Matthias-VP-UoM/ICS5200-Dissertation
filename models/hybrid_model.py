import os
import psutil
import time
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
import optuna

from models.cnn_model_v2 import build_resnet18
from models.vit_model_v2 import build_vit
from utils.image_loader import WebpageScreenshotDataset, get_image_transforms
from utils.evaluation import evaluate_model, plot_confusion_matrix, save_classification_report


SEED = 42
SCORING_METHOD = "f1_weighted"
N_TRIALS = 100

def log_resource_usage(stage_name=""):
    """Prints the current CPU and RAM usage of the Python process."""
    process = psutil.Process(os.getpid())
    
    # RAM usage in Megabytes
    ram_usage_mb = process.memory_info().rss / (1024 * 1024) 
    
    # CPU usage percentage (interval calculates usage over 0.1 seconds)
    cpu_usage_pct = psutil.cpu_percent(interval=1)
    
    print(f"--- [Resource Log: {stage_name}] ---")
    print(f"RAM Usage: {ram_usage_mb:.2f} MB")
    print(f"System CPU Usage: {cpu_usage_pct}%")
    print("-" * 35)


def extract_embeddings(df, backbone="cnn", batch_size=16, num_classes=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = get_image_transforms(224, is_train=False)
    dataset = WebpageScreenshotDataset(df, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if backbone == "cnn":
        model = build_resnet18(num_classes=num_classes)
        model.load_state_dict(torch.load("outputs/models/cnn_production_final_2.pth"))
        model.fc = torch.nn.Identity()

    elif backbone == "vit":
        model = build_vit(num_classes=num_classes)
        model.load_state_dict(torch.load("outputs/models/vit_production_final_2.pth"))
        model.heads.head = torch.nn.Identity()

    else:
        raise ValueError("backbone must be either 'cnn' or 'vit'")

    model = model.to(device)
    model.eval()

    embeddings = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            features = model(images)
            embeddings.append(features.cpu().numpy())

    return np.vstack(embeddings)


def build_hybrid_features(train_df, test_df, tabular_feature_columns, backbone="cnn"):
    print(f"\nExtracting {backbone.upper()} embeddings...")

    train_embeddings = extract_embeddings(train_df, backbone=backbone)
    test_embeddings = extract_embeddings(test_df, backbone=backbone)

    X_train_tabular = train_df[tabular_feature_columns].values
    X_test_tabular = test_df[tabular_feature_columns].values

    scaler = StandardScaler()
    X_train_tabular_scaled = scaler.fit_transform(X_train_tabular)
    X_test_tabular_scaled = scaler.transform(X_test_tabular)

    X_train_hybrid = np.hstack([train_embeddings, X_train_tabular_scaled])
    X_test_hybrid = np.hstack([test_embeddings, X_test_tabular_scaled])

    y_train = train_df["accessibility_label"].astype(int).values
    y_test = test_df["accessibility_label"].astype(int).values

    hybrid_feature_names = (
        [f"{backbone}_embedding_{i}" for i in range(train_embeddings.shape[1])]
        + tabular_feature_columns
    )

    return X_train_hybrid, X_test_hybrid, y_train, y_test, hybrid_feature_names, scaler


def tune_hybrid_gradient_boosting(X_train, y_train, n_trials=N_TRIALS):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300]),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_samples_split": trial.suggest_int("min_samples_split", 10, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }

        model = GradientBoostingClassifier(
            **params,
            random_state=SEED
        )

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=SCORING_METHOD,
            n_jobs=-1
        )

        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    os.makedirs("outputs/results", exist_ok=True)
    study.trials_dataframe().to_csv(
        "outputs/results/hybrid_gb_optuna_trials.csv",
        index=False
    )

    print("Best Hybrid GB parameters:", study.best_params)
    return study.best_params


def train_hybrid_gradient_boosting(train_df, test_df, tabular_feature_columns, backbone="cnn", use_optuna_tuning=True):
    start_time = time.time()

    X_train, X_test, y_train, y_test, hybrid_feature_names, scaler = build_hybrid_features(
        train_df=train_df,
        test_df=test_df,
        tabular_feature_columns=tabular_feature_columns,
        backbone=backbone
    )

    if use_optuna_tuning:
        best_params = tune_hybrid_gradient_boosting(X_train, y_train)
    else:
        best_params = {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 4,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "max_features": "sqrt"
        }

    model = GradientBoostingClassifier(
        **best_params,
        random_state=SEED
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results = evaluate_model(
        f"Hybrid {backbone.upper()} + Gradient Boosting",
        y_test,
        predictions
    )

    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)

    joblib.dump(model, f"outputs/models/hybrid_{backbone}_gb.pkl")
    joblib.dump(scaler, f"outputs/models/hybrid_{backbone}_tabular_scaler.pkl")
    joblib.dump(hybrid_feature_names, f"outputs/models/hybrid_{backbone}_feature_names.pkl")

    plot_confusion_matrix(
        f"Hybrid {backbone.upper()} + Gradient Boosting",
        y_test,
        predictions,
        f"outputs/plots/hybrid_{backbone}_gb_confusion_matrix.png"
    )

    save_classification_report(
        y_test,
        predictions,
        f"outputs/results/hybrid_{backbone}_gb_class_report.csv"
    )

    print(f"Hybrid training time: {time.time() - start_time:.2f} seconds")

    return model, results


def tune_hybrid_random_forest(X_train, y_train, n_trials=N_TRIALS):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_categorical("n_estimators", [200, 300, 500]),
            # "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2"]
            ),
            "max_samples": trial.suggest_categorical("max_samples", [0.6, 0.8, 1.0]),
        }

        model = RandomForestClassifier(
            random_state=42,
            class_weight="balanced_subsample",
            # n_jobs=-1,
            bootstrap = True,
            **params
        )

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=SCORING_METHOD,
            n_jobs=-1
        )

        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    os.makedirs("outputs/results", exist_ok=True)
    study.trials_dataframe().to_csv(
        "outputs/results/hybrid_rf_optuna_trials.csv",
        index=False
    )

    print("Best Hybrid RF parameters:", study.best_params)
    return study.best_params


def train_hybrid_random_forest(train_df, test_df, tabular_feature_columns, backbone="cnn", use_optuna_tuning=True):
    X_train, X_test, y_train, y_test, hybrid_feature_names, scaler = build_hybrid_features(
        train_df=train_df,
        test_df=test_df,
        tabular_feature_columns=tabular_feature_columns,
        backbone=backbone
    )

    if use_optuna_tuning:
        best_params = tune_hybrid_gradient_boosting(X_train, y_train)
    else:
        best_params = {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 4,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "max_features": "sqrt"
        }

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=17,
        min_samples_split=3,
        min_samples_leaf=3,
        max_features="sqrt",
        max_samples=0.6,
        bootstrap=True,
        class_weight="balanced_subsample",
        random_state=SEED
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results = evaluate_model(
        f"Hybrid {backbone.upper()} + Random Forest",
        y_test,
        predictions
    )

    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)

    joblib.dump(model, f"outputs/models/hybrid_{backbone}_rf.pkl")
    joblib.dump(scaler, f"outputs/models/hybrid_{backbone}_tabular_scaler.pkl")
    joblib.dump(hybrid_feature_names, f"outputs/models/hybrid_{backbone}_feature_names.pkl")

    plot_confusion_matrix(
        f"Hybrid {backbone.upper()} + Random Forest",
        y_test,
        predictions,
        f"outputs/plots/hybrid_{backbone}_rf_confusion_matrix.png"
    )

    save_classification_report(
        y_test,
        predictions,
        f"outputs/results/hybrid_{backbone}_rf_class_report.csv"
    )

    return model, results