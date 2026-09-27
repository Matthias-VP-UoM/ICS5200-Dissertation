import optuna
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
import os
import time
import psutil

from utils.evaluation import (
    evaluate_model,
    save_feature_importance,
    plot_confusion_matrix,
    save_classification_report,
    generate_global_shap_plots,
    generate_shap_error_plots,
    compare_global_vs_error_shap,
    generate_per_class_beeswarm
)

SCORING_METHOD = "f1_weighted"
N_TRIALS = 100
SEED = 42

# Utility function to print out memory (RAM) and processing (CPU) usage
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

# Function which optimises the hyperparameters of the Gradient Boosting model using Optuna
def tune_gradboost(X_train, y_train, n_trials=N_TRIALS):
    CV_VAL = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    def objective(trial):
        model_hyperparams = {
            "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300]),
            # "learning_rate": trial.suggest_categorical("learning_rate", [1.0, 0.1, 0.01]),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_samples_split": trial.suggest_int("min_samples_split", 10, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 20),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            # "max_samples": trial.suggest_categorical("max_samples", [0.6, 0.8, 1.0]),
            # "bootstrap": True,
        }

        classifier = GradientBoostingClassifier(
            random_state=42,
            # class_weight="balanced_subsample",
            # n_jobs=-1,
            **model_hyperparams,
        )

        # Utilise sampling in order to increase the number of instances for the non-majority classes
        # This is done to provide balanced training, ensuring that no class is deemed more important
        # than another
        model = Pipeline([
            (
                "oversampler",
                RandomOverSampler(
                    sampling_strategy="not majority",
                    random_state=SEED
                )
            ),
            (
                "classifier",
                classifier
            )
        ])

        # model = GradientBoostingClassifier(
        #     n_estimators=trial.suggest_categorical("n_estimators", [100, 200, 300]),
        #     # learning_rate=trial.suggest_categorical("learning_rate", [1.0, 0.1, 0.01]),
        #     learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        #     max_depth=trial.suggest_int("max_depth", 2, 6),
        #     min_samples_split=trial.suggest_int("min_samples_split", 10, 30),
        #     min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 20),
        #     max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        #     # subsample=trial.suggest_float("subsample", 0.6, 1.0),
        #     # colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        #     # gamma=trial.suggest_float("gamma", 0.0, 5.0),
        #     # reg_alpha=trial.suggest_float("reg_alpha", 0.0, 5.0),
        #     # reg_lambda=trial.suggest_float("reg_lambda", 0.0, 5.0),
        #     # objective="multi:softmax",
        #     # num_class=len(np.unique(y_train)),
        #     # class_weight="balanced_subsample",
        #     random_state=42,
        #     # n_jobs=-1
        # )

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=CV_VAL,
            scoring=SCORING_METHOD,
            n_jobs=-1
        )

        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    study.trials_dataframe().to_csv(
        "outputs/results/gb_optuna_trials.csv",
        # "outputs/results/gb_optuna_trials_v7.csv",
        index=False
    )

    print("Best Gradient Boosting parameters:", study.best_params)
    return study.best_params

# Function used to train the Gradient Boosting model
def train_grad_boost(X_train, X_test, y_train, y_test, feature_columns, feature_group_name=None, use_optuna_tuning=True):
    print("\n==============================")
    print("TRAINING GRADIENT BOOSTING")
    print("==============================")

    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

    log_resource_usage("Before Tuning/Training")

    if use_optuna_tuning:
        # Optimise the hyperparameters
        start_time = time.time()
        best_params = tune_gradboost(X_train, y_train)
        end_time = time.time()

        tune_time = end_time - start_time
        print('Time taken to optimise the Gradient Boosting model:', tune_time)
        log_resource_usage("After Optuna Tuning")
    else:
        # Use the best recorded hyperparameters if optimisation is not needed
        best_params = {
            "n_estimators": 300,
            "learning_rate": 0.0357656967477886,
            "max_depth": 6,
            "min_samples_split": 14,
            "min_samples_leaf": 18,
            "max_features": "log2"
        }
        # best_params = {
        #     "n_estimators": 200,
        #     "learning_rate": 0.061401859587171786,
        #     "max_depth": 5,
        #     "min_samples_split": 15,
        #     "min_samples_leaf": 19,
        #     "max_features": "sqrt"
        # }

    classifier = GradientBoostingClassifier(
        **best_params,
        random_state=42,
        # eval_metric="mlogloss"
    )
    # formerly model = GradientBoostingClassifier

    model = Pipeline([
        # (
        #     "oversampler",
        #     RandomOverSampler(
        #         sampling_strategy="not majority",
        #         random_state=SEED
        #     )
        # ),
        (
            "classifier",
            classifier
        )
    ])

    model.fit(X_train, y_train)
    log_resource_usage("After Final Model Fit")

    predictions = model.predict(X_test)

    results = evaluate_model(
        "Gradient Boosting",
        y_test,
        predictions
    )

    trained_classifier = model.named_steps["classifier"]

    # Save the feature importance for the model
    if feature_group_name is None:
        save_feature_importance(
            trained_classifier,
            feature_columns,
            "outputs/feature_importance/gb_feature_importance.csv"
            # "outputs/feature_importance/gb_feature_importance_v7.csv"
        )
    else:
        save_feature_importance(
            trained_classifier,
            feature_columns,
            f"outputs/feature_importance/gb_feature_importance_{feature_group_name}.csv"
        )
    
    # Plot and save the model's confusion matrix
    if feature_group_name is None:
        plot_confusion_matrix(
            "Gradient Boosting",
            y_test,
            predictions,
            "outputs/plots/gb_confusion_matrix.png"
            # "outputs/plots/gb_confusion_matrix_v7.png"
        )
    else:
        plot_confusion_matrix(
            "Gradient Boosting",
            y_test,
            predictions,
            f"outputs/plots/gb_confusion_matrix_{feature_group_name}.png"
        )
    
    # Save the model's classification report
    if feature_group_name is None:
        save_classification_report(
            y_test,
            predictions,
            "outputs/results/gb_class_report.csv"
            # "outputs/results/gb_class_report_v7.csv"
        )
    else:
        save_classification_report(
            y_test,
            predictions,
            f"outputs/results/gb_class_report_{feature_group_name}.csv"
        )
    
    # Generate and save the global SHAP plots of the model
    if feature_group_name is None:
        shap_values = generate_global_shap_plots(
            trained_classifier,
            X_test,
            feature_columns,
            "Gradient Boosting"
        )
    else:
        shap_values = generate_global_shap_plots(
            trained_classifier,
            X_test,
            feature_columns,
            f"Gradient Boosting - {feature_group_name}"
        )
    
    # Generate and save the beeswarm plot from the generated SHAP values
    if feature_group_name is None:
        top_features = generate_per_class_beeswarm(
            shap_values,
            X_test=X_test,
            feature_columns=feature_columns,
            model_name="Gradient Boosting",
            class_names=["High", "Medium", "Low"],  # replace with your actual class labels
            top_n=10
        )
    else:
        top_features = generate_per_class_beeswarm(
            shap_values,
            X_test=X_test,
            feature_columns=feature_columns,
            model_name=f"Gradient Boosting - {feature_group_name}",
            class_names=["High", "Medium", "Low"],  # replace with your actual class labels
            top_n=10
        )
    
    # Generate and save the SHAP error plots
    if feature_group_name is None:
        generate_shap_error_plots(
            trained_classifier,
            X_test,
            y_test,
            predictions,
            feature_columns,
            "Gradient Boosting",
            shap_values
        )
    else:
        generate_shap_error_plots(
            trained_classifier,
            X_test,
            y_test,
            predictions,
            feature_columns,
            f"Gradient Boosting - {feature_group_name}",
            shap_values
        )
    
    if feature_group_name is None:
        compare_global_vs_error_shap(
            shap_values,
            X_test,
            y_test,
            predictions,
            feature_columns,
            "outputs/results/gb_global_vs_error_shap_comparison.csv"
            # "outputs/results/gb_global_vs_error_shap_comparison_v7.csv"
        )
    else:
        compare_global_vs_error_shap(
            shap_values,
            X_test,
            y_test,
            predictions,
            feature_columns,
            f"outputs/results/gb_global_vs_error_shap_comparison_{feature_group_name}.csv"
        )

    return model, results