# ============================================================
# Common hybrid experiment configuration
# ============================================================
from pathlib import Path
import copy
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
import optuna

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import models

from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_sample_weight

from imblearn.over_sampling import RandomOverSampler


from utils.image_loader import WebpageScreenshotDataset, get_image_transforms, IMG_SIZE
from utils.data_loader import FEATURES_FOR_SCORING
from utils.evaluation import evaluate_model, plot_confusion_matrix, save_classification_report

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = Path("/content/data")
DATASET_PATH = DATA_DIR / "annotated_dataset.csv"
TRAIN_SPLIT_PATH = DATA_DIR / "train_split.csv"
TEST_SPLIT_PATH = DATA_DIR / "test_split.csv"

MODEL_DIR = Path("outputs/models")
RESULT_DIR = Path("outputs/results")
PLOT_DIR = Path("outputs/plots")
EMBEDDING_DIR = Path("outputs/embeddings")

for directory in [MODEL_DIR, RESULT_DIR, PLOT_DIR, EMBEDDING_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 3
CLASS_NAMES = ["High", "Medium", "Low"]
SEED = 42

# ------------------------------------------------------------
# Supported visual architectures and trained checkpoints
# ------------------------------------------------------------
CNN_RN18_CHECKPOINT = MODEL_DIR / "cnn_production_final_resnet18.pth"
CNN_RN50_CHECKPOINT = MODEL_DIR / "cnn_production_final_resnet50.pth"
CNN_VGG16_CHECKPOINT = MODEL_DIR / "cnn_production_final_vgg16.pth"

VIT_B_16_CHECKPOINT = MODEL_DIR / "vit_b_16_production_final.pth"
VIT_SWIN_T_CHECKPOINT = MODEL_DIR / "swin_t_production_final.pth"

ARCHITECTURE_CONFIG = {
    "resnet18": {
        "family": "cnn",
        "checkpoint": CNN_RN18_CHECKPOINT,
        "embedding_dim": 512,
    },
    "resnet50": {
        "family": "cnn",
        "checkpoint": CNN_RN50_CHECKPOINT,
        "embedding_dim": 2048,
    },
    "vgg16": {
        "family": "cnn",
        "checkpoint": CNN_VGG16_CHECKPOINT,
        "embedding_dim": 4096,
    },
    "vit_b_16": {
        "family": "vit",
        "checkpoint": VIT_B_16_CHECKPOINT,
        "embedding_dim": 768,
    },
    "swin_t": {
        "family": "vit",
        "checkpoint": VIT_SWIN_T_CHECKPOINT,
        "embedding_dim": 768,
    },
}

SUPPORTED_ARCHITECTURES = tuple(ARCHITECTURE_CONFIG.keys())


def _normalise_architecture_name(architecture):
    aliases = {
        "rn18": "resnet18",
        "resnet_18": "resnet18",
        "rn50": "resnet50",
        "resnet_50": "resnet50",
        "vgg": "vgg16",
        "vit": "vit_b_16",
        "vit_b16": "vit_b_16",
        "vit-b-16": "vit_b_16",
        "swin": "swin_t",
        "swin-t": "swin_t",
    }

    architecture = str(architecture).strip().lower()
    architecture = aliases.get(architecture, architecture)

    if architecture not in ARCHITECTURE_CONFIG:
        raise ValueError(
            f"Unsupported architecture '{architecture}'. "
            f"Choose one of: {SUPPORTED_ARCHITECTURES}"
        )

    return architecture


def _unwrap_state_dict(checkpoint):
    """
    Accept either a raw state_dict or a checkpoint dictionary containing
    model_state_dict/state_dict/model keys.
    """
    if not isinstance(checkpoint, dict):
        return checkpoint

    for key in ("model_state_dict", "state_dict", "model"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]

    return checkpoint


def _remove_module_prefix(state_dict):
    """
    Handle checkpoints produced using DataParallel/DistributedDataParallel.
    """
    if any(key.startswith("module.") for key in state_dict):
        return {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
    return state_dict


def _checkpoint_uses_dropout_head(state_dict, architecture):
    """
    Detect whether the saved classifier was:
        Linear(...)
    or:
        Sequential(Dropout(...), Linear(...))

    This allows the original trained architecture to be rebuilt before its
    checkpoint is loaded.
    """
    sequential_weight_keys = {
        "resnet18": "fc.1.weight",
        "resnet50": "fc.1.weight",
        "vgg16": "classifier.6.1.weight",
        "vit_b_16": "heads.head.1.weight",
        "swin_t": "head.1.weight",
    }
    return sequential_weight_keys[architecture] in state_dict


def build_classification_architecture(
    architecture,
    num_classes=NUM_CLASSES,
    dropout=0.0,
    force_sequential_head=False,
):
    """
    Construct any supported standalone image classifier.

    `force_sequential_head=True` rebuilds the final head as
    Sequential(Dropout, Linear), matching tuned checkpoints that used dropout.
    """
    architecture = _normalise_architecture_name(architecture)

    if architecture == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = (
            nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
            if force_sequential_head
            else nn.Linear(in_features, num_classes)
        )

    elif architecture == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = (
            nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
            if force_sequential_head
            else nn.Linear(in_features, num_classes)
        )

    elif architecture == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        in_features = model.classifier[6].in_features
        model.classifier[6] = (
            nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
            if force_sequential_head
            else nn.Linear(in_features, num_classes)
        )

    elif architecture == "vit_b_16":
        # model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model = timm.create_model('vit_base_patch16_224', pretrained=False, dynamic_img_size=True)
        in_features = model.head.in_features
        model.head = (
            nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
            if force_sequential_head
            else nn.Linear(in_features, num_classes)
        )

    elif architecture == "swin_t":
        model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        in_features = model.head.in_features
        model.head = (
            nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
            if force_sequential_head
            else nn.Linear(in_features, num_classes)
        )

    return model


def replace_classification_head_with_identity(model, architecture):
    """
    Remove only the final class-prediction layer after loading the complete
    trained checkpoint, so the network returns its penultimate embedding.
    """
    architecture = _normalise_architecture_name(architecture)

    if architecture in {"resnet18", "resnet50"}:
        model.fc = nn.Identity()
    elif architecture == "vgg16":
        model.classifier[6] = nn.Identity()
    elif architecture == "vit_b_16":
        model.head = nn.Identity()
    elif architecture == "swin_t":
        model.head = nn.Identity()

    return model


def load_master_splits():
    """Load the exact persistent split used by all standalone models."""
    df = pd.read_csv(DATASET_PATH)
    train_df = pd.read_csv(TRAIN_SPLIT_PATH)
    test_df = pd.read_csv(TEST_SPLIT_PATH)

    for frame in [df, train_df, test_df]:
        frame["id"] = frame["id"].astype(str).str.zfill(3)
        frame["screenshot_path"] = (
            frame["screenshot_path"]
            .astype(str)
            .str.replace("\\", "/", regex=False)
        )

    train_ids = set(train_df["id"])
    test_ids = set(test_df["id"])

    # Rebuild from the master dataset so the latest feature columns are used,
    # while preserving the exact original split membership.
    train_df = df[df["id"].isin(train_ids)].copy()
    test_df = df[df["id"].isin(test_ids)].copy()

    assert set(train_df["id"]).isdisjoint(set(test_df["id"]))
    assert len(train_df) + len(test_df) <= len(df)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print(
        "Train labels:\n",
        train_df["accessibility_label"].value_counts().sort_index(),
    )
    print(
        "Test labels:\n",
        test_df["accessibility_label"].value_counts().sort_index(),
    )
    return train_df, test_df


def load_finetuned_backbone(
    architecture,
    checkpoint_path=None,
    dropout=0.0,
):
    """
    Load a task-specific standalone checkpoint and convert the trained model
    into a fixed embedding extractor.

    The complete classifier is reconstructed and loaded first. Its final
    prediction layer is replaced with Identity only after successful loading.
    """
    architecture = _normalise_architecture_name(architecture)
    config = ARCHITECTURE_CONFIG[architecture]

    checkpoint_path = Path(
        checkpoint_path
        if checkpoint_path is not None
        else config["checkpoint"]
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found for {architecture}: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )
    state_dict = _remove_module_prefix(_unwrap_state_dict(checkpoint))

    if architecture == "vit_b_16" and "pos_embed" in state_dict:
        checkpoint_pos_embed = state_dict["pos_embed"] # Shape: [1, 881, 768]

        # We need to compute the grid size for the checkpoint (881 - 1 CLS token = 880 patches)
        # 880 isn't a perfect square, which implies a non-square image aspect ratio was used.
        # Let's dynamically resize it to the expected 197 tokens (14x14 grid + 1 CLS token)
        original_grid = [20, 44]

        from timm.layers import resample_abs_pos_embed

        # Interpolate the checkpoint's 881 pos_embed to match the model's target sequence length of 197
        state_dict["pos_embed"] = resample_abs_pos_embed(
            checkpoint_pos_embed,
            new_size=[14, 14],  # Target grid size for 224x224 image with 16x16 patches
            old_size=original_grid,   # Tells timm precisely how to reshape the 880 tokens
            num_prefix_tokens=1
        )

    uses_dropout_head = _checkpoint_uses_dropout_head(
        state_dict,
        architecture,
    )

    model = build_classification_architecture(
        architecture=architecture,
        num_classes=NUM_CLASSES,
        dropout=dropout,
        force_sequential_head=uses_dropout_head,
    )

    model.load_state_dict(state_dict, strict=True)
    model = replace_classification_head_with_identity(
        model,
        architecture,
    )

    model = model.to(DEVICE)
    model.eval()

    embedding_dim = config["embedding_dim"]
    print(
        f"Loaded {architecture} as a fine-tuned feature extractor "
        f"({embedding_dim}-dimensional embedding)."
    )
    return model, embedding_dim


def extract_embeddings(
    frame,
    architecture,
    split_name,
    checkpoint_path=None,
    batch_size=8,
    use_cache=True,
    dropout=0.0,
):
    architecture = _normalise_architecture_name(architecture)

    cache_path = (
        EMBEDDING_DIR
        / f"{split_name}_{architecture}_finetuned_embeddings.npy"
    )
    id_path = (
        EMBEDDING_DIR
        / f"{split_name}_{architecture}_ids.npy"
    )

    if use_cache and cache_path.exists() and id_path.exists():
        cached_ids = np.load(
            id_path,
            allow_pickle=True,
        ).astype(str)
        current_ids = frame["id"].astype(str).to_numpy()

        if np.array_equal(cached_ids, current_ids):
            print(
                f"Loading cached {architecture} embeddings:",
                cache_path,
            )
            return np.load(cache_path)

        print(
            f"Cached IDs do not match the current {split_name} split. "
            "Embeddings will be regenerated."
        )

    transform = get_image_transforms(
        IMG_SIZE,
        is_train=False,
    )
    dataset = WebpageScreenshotDataset(
        frame,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model, expected_embedding_dim = load_finetuned_backbone(
        architecture=architecture,
        checkpoint_path=checkpoint_path,
        dropout=dropout,
    )

    collected = []

    with torch.inference_mode():
        for images, _ in loader:
            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            with torch.amp.autocast(
                device_type=DEVICE.type,
                enabled=(DEVICE.type == "cuda"),
            ):
                batch_embeddings = model(images)

            collected.append(
                batch_embeddings.detach().cpu().numpy()
            )

    embeddings = np.vstack(collected)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional embedding matrix, "
            f"but received shape {embeddings.shape}."
        )

    if embeddings.shape[1] != expected_embedding_dim:
        raise ValueError(
            f"{architecture} should produce {expected_embedding_dim} "
            f"features, but produced {embeddings.shape[1]}."
        )

    np.save(cache_path, embeddings)
    np.save(
        id_path,
        frame["id"].astype(str).to_numpy(),
    )

    print(
        f"Saved {architecture} embeddings:",
        cache_path,
        embeddings.shape,
    )
    return embeddings


def prepare_tabular(train_df, test_df):
    missing = [
        column
        for column in FEATURES_FOR_SCORING
        if column not in train_df.columns
    ]
    if missing:
        raise ValueError(
            f"Missing tabular features: {missing}"
        )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        train_df[FEATURES_FOR_SCORING]
    )
    X_test = scaler.transform(
        test_df[FEATURES_FOR_SCORING]
    )

    y_train = (
        train_df["accessibility_label"]
        .astype(int)
        .to_numpy()
    )
    y_test = (
        test_df["accessibility_label"]
        .astype(int)
        .to_numpy()
    )

    return X_train, X_test, y_train, y_test, scaler


def evaluate_and_save(
    name,
    model,
    X_test,
    y_test,
):
    predictions = model.predict(X_test)
    results = evaluate_model(
        name,
        y_test,
        predictions,
    )

    safe_name = (
        name.lower()
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("/", "_")
    )

    plot_confusion_matrix(
        name,
        y_test,
        predictions,
        PLOT_DIR / f"{safe_name}_cm.png",
    )
    save_classification_report(
        y_test,
        predictions,
        RESULT_DIR / f"{safe_name}_report.csv",
    )

    return predictions, results

# ============================================================
# Hybrid feature-fusion preprocessing and training
# ============================================================
def prepare_fused_features(
    train_df,
    test_df,
    train_embeddings,
    test_embeddings,
    *,
    scale_tabular=True,
    scale_embeddings=True,
    use_pca=False,
    pca_variance=0.95,
):
    """
    Prepare the two branches independently and concatenate them afterwards.

    Processing order
    ----------------
    1. Read the original handcrafted features.
    2. Fit the tabular scaler on the training split only.
    3. Fit a separate embedding scaler on the training split only.
    4. Optionally fit PCA on the scaled training embeddings only.
    5. Concatenate visual and tabular branches row by row.

    The test split is transformed with the training-fitted artefacts and is
    never used when fitting a scaler or PCA model.
    """
    if len(train_df) != len(train_embeddings):
        raise ValueError(
            "Training row mismatch: "
            f"{len(train_df)} dataframe rows versus "
            f"{len(train_embeddings)} embedding rows."
        )
    if len(test_df) != len(test_embeddings):
        raise ValueError(
            "Test row mismatch: "
            f"{len(test_df)} dataframe rows versus "
            f"{len(test_embeddings)} embedding rows."
        )

    missing = [
        feature
        for feature in FEATURES_FOR_SCORING
        if feature not in train_df.columns or feature not in test_df.columns
    ]
    if missing:
        raise ValueError(f"Missing tabular features: {sorted(set(missing))}")

    X_train_tabular_raw = (
        train_df[FEATURES_FOR_SCORING]
        .apply(pd.to_numeric, errors="raise")
        .to_numpy(dtype=np.float32)
    )
    X_test_tabular_raw = (
        test_df[FEATURES_FOR_SCORING]
        .apply(pd.to_numeric, errors="raise")
        .to_numpy(dtype=np.float32)
    )

    X_train_embeddings_raw = np.asarray(train_embeddings, dtype=np.float32)
    X_test_embeddings_raw = np.asarray(test_embeddings, dtype=np.float32)

    y_train = train_df["accessibility_label"].astype(int).to_numpy()
    y_test = test_df["accessibility_label"].astype(int).to_numpy()

    # Fit one scaler per modality. This keeps the visual and handcrafted
    # branches independent until the fusion step.
    tabular_scaler = StandardScaler() if scale_tabular else None
    embedding_scaler = StandardScaler() if scale_embeddings else None

    if tabular_scaler is not None:
        X_train_tabular = tabular_scaler.fit_transform(X_train_tabular_raw)
        X_test_tabular = tabular_scaler.transform(X_test_tabular_raw)
    else:
        X_train_tabular = X_train_tabular_raw.copy()
        X_test_tabular = X_test_tabular_raw.copy()

    if embedding_scaler is not None:
        X_train_visual = embedding_scaler.fit_transform(
            X_train_embeddings_raw
        )
        X_test_visual = embedding_scaler.transform(
            X_test_embeddings_raw
        )
    else:
        X_train_visual = X_train_embeddings_raw.copy()
        X_test_visual = X_test_embeddings_raw.copy()

    embedding_pca = None
    visual_dim_before_pca = X_train_visual.shape[1]

    if use_pca:
        if not 0.0 < pca_variance <= 1.0:
            raise ValueError("pca_variance must be within (0, 1].")

        embedding_pca = PCA(
            n_components=pca_variance,
            svd_solver="full",
            random_state=SEED,
        )
        X_train_visual = embedding_pca.fit_transform(X_train_visual)
        X_test_visual = embedding_pca.transform(X_test_visual)

    X_train_fused = np.concatenate(
        [X_train_visual, X_train_tabular],
        axis=1,
    ).astype(np.float32, copy=False)
    X_test_fused = np.concatenate(
        [X_test_visual, X_test_tabular],
        axis=1,
    ).astype(np.float32, copy=False)

    visual_feature_names = [
        f"visual_component_{index:04d}"
        for index in range(X_train_visual.shape[1])
    ]
    fused_feature_names = visual_feature_names + list(FEATURES_FOR_SCORING)

    print("\n" + "=" * 68)
    print("HYBRID FEATURE-FUSION PREPROCESSING")
    print("=" * 68)
    print(f"Training rows:                    {len(y_train)}")
    print(f"Test rows:                        {len(y_test)}")
    print(f"Visual dimensions before PCA:     {visual_dim_before_pca}")
    print(f"Visual dimensions after PCA:      {X_train_visual.shape[1]}")
    print(f"Tabular dimensions:               {X_train_tabular.shape[1]}")
    print(f"Final fused dimensions:           {X_train_fused.shape[1]}")
    print(f"Separate tabular scaling:         {scale_tabular}")
    print(f"Separate embedding scaling:       {scale_embeddings}")
    print(f"Embedding PCA enabled:            {use_pca}")
    if embedding_pca is not None:
        print(
            "PCA variance retained:            "
            f"{embedding_pca.explained_variance_ratio_.sum():.4f}"
        )
    print("Training class distribution:")
    print(pd.Series(y_train).value_counts().sort_index().to_string())
    print("=" * 68 + "\n")

    return {
        "X_train_fused": X_train_fused,
        "X_test_fused": X_test_fused,
        "X_train_visual": X_train_visual,
        "X_test_visual": X_test_visual,
        "X_train_tabular": X_train_tabular,
        "X_test_tabular": X_test_tabular,
        "y_train": y_train,
        "y_test": y_test,
        "tabular_scaler": tabular_scaler,
        "embedding_scaler": embedding_scaler,
        "embedding_pca": embedding_pca,
        "fused_feature_names": fused_feature_names,
        "visual_dim": X_train_visual.shape[1],
        "tabular_dim": X_train_tabular.shape[1],
    }


def balance_fused_training_data(
    X_train_fused,
    y_train,
    *,
    strategy="oversample",
):
    """
    Apply class-imbalance handling only after the training branches have been
    aligned, transformed and fused.

    Supported strategies
    --------------------
    none:
        Leave the training rows unchanged.
    oversample:
        Duplicate minority-class fused rows with RandomOverSampler.
    sample_weight:
        Keep the rows unchanged and return balanced per-row weights. This is
        suitable for GradientBoostingClassifier.fit(sample_weight=...).

    The test set must never be passed to this function.
    """
    strategy = str(strategy).strip().lower()
    valid = {"none", "oversample", "sample_weight"}
    if strategy not in valid:
        raise ValueError(f"strategy must be one of {sorted(valid)}")

    X_train_fused = np.asarray(X_train_fused)
    y_train = np.asarray(y_train)

    before = pd.Series(y_train).value_counts().sort_index()
    sample_weight = None
    sampler = None

    if strategy == "oversample":
        sampler = RandomOverSampler(
            sampling_strategy="not majority",
            random_state=SEED,
        )
        X_balanced, y_balanced = sampler.fit_resample(
            X_train_fused,
            y_train,
        )
    elif strategy == "sample_weight":
        X_balanced = X_train_fused
        y_balanced = y_train
        sample_weight = compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )
    else:
        X_balanced = X_train_fused
        y_balanced = y_train

    after = pd.Series(y_balanced).value_counts().sort_index()

    print("Class-imbalance strategy:", strategy)
    print("Before:")
    print(before.to_string())
    print("After:")
    print(after.to_string())
    if sample_weight is not None:
        print(
            "Sample-weight range: "
            f"{sample_weight.min():.4f} to {sample_weight.max():.4f}"
        )
    print()

    return {
        "X_train": X_balanced,
        "y_train": y_balanced,
        "sample_weight": sample_weight,
        "sampler": sampler,
    }


def build_hybrid_classifier(
    estimator_type="rf",
    *,
    rf_class_weight=None,
    model_params=None,
):
    """Create the downstream classifier used after feature fusion."""
    estimator_type = str(estimator_type).strip().lower()
    model_params = dict(model_params or {})

    if estimator_type == "rf":
        defaults = {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": rf_class_weight,
            "random_state": SEED,
            "n_jobs": -1,
        }
        defaults.update(model_params)
        return RandomForestClassifier(**defaults)

    if estimator_type == "gb":
        defaults = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 3,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "subsample": 1.0,
            "random_state": SEED,
        }
        defaults.update(model_params)
        return GradientBoostingClassifier(**defaults)

    raise ValueError("estimator_type must be either 'rf' or 'gb'.")


def _suggest_hybrid_parameters(trial, estimator_type):
    """Define the Optuna search space for the downstream fusion classifier."""
    estimator_type = str(estimator_type).strip().lower()

    if estimator_type == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 600, step=50),
            "max_depth": trial.suggest_categorical(
                "max_depth", [None, 8, 12, 16, 24, 32]
            ),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.1, 0.2, 0.4, 0.6]
            ),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }

    if estimator_type == "gb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical(
                "max_features", [None, "sqrt", "log2", 0.1, 0.2, 0.4]
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
        }

    raise ValueError("estimator_type must be either 'rf' or 'gb'.")


def tune_hybrid_estimator(
    train_df,
    train_embeddings,
    *,
    estimator_type="rf",
    imbalance_strategy="oversample",
    scale_tabular=True,
    scale_embeddings=True,
    use_pca=False,
    pca_variance=0.95,
    n_trials=40,
    n_splits=5,
):
    """
    Tune the downstream RF/GB classifier using leakage-safe stratified CV.

    For every fold, the tabular scaler, embedding scaler and optional PCA are
    fitted only on that fold's training rows. Oversampling/sample weighting is
    also applied only to the fold's training rows.
    """
    estimator_type = str(estimator_type).strip().lower()
    y_all = train_df["accessibility_label"].astype(int).to_numpy()
    embeddings_all = np.asarray(train_embeddings)

    if len(train_df) != len(embeddings_all):
        raise ValueError("train_df and train_embeddings are not row-aligned.")

    rf_class_weight = None
    effective_strategy = imbalance_strategy
    if estimator_type == "rf" and imbalance_strategy in {
        "balanced",
        "balanced_subsample",
    }:
        rf_class_weight = imbalance_strategy
        effective_strategy = "none"

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED,
    )

    def objective(trial):
        params = _suggest_hybrid_parameters(trial, estimator_type)
        fold_scores = []

        for fold_number, (fit_idx, val_idx) in enumerate(
            cv.split(np.zeros(len(y_all)), y_all),
            start=1,
        ):
            fold_train_df = train_df.iloc[fit_idx].reset_index(drop=True)
            fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)
            fold_train_embeddings = embeddings_all[fit_idx]
            fold_val_embeddings = embeddings_all[val_idx]

            fold_prepared = prepare_fused_features(
                train_df=fold_train_df,
                test_df=fold_val_df,
                train_embeddings=fold_train_embeddings,
                test_embeddings=fold_val_embeddings,
                scale_tabular=scale_tabular,
                scale_embeddings=scale_embeddings,
                use_pca=use_pca,
                pca_variance=pca_variance,
            )

            fold_balanced = balance_fused_training_data(
                fold_prepared["X_train_fused"],
                fold_prepared["y_train"],
                strategy=effective_strategy,
            )

            model = build_hybrid_classifier(
                estimator_type=estimator_type,
                rf_class_weight=rf_class_weight,
                model_params=params,
            )

            fit_kwargs = {}
            if fold_balanced["sample_weight"] is not None:
                fit_kwargs["sample_weight"] = fold_balanced["sample_weight"]

            model.fit(
                fold_balanced["X_train"],
                fold_balanced["y_train"],
                **fit_kwargs,
            )
            predictions = model.predict(fold_prepared["X_test_fused"])
            score = f1_score(
                fold_prepared["y_test"],
                predictions,
                average="macro",
                zero_division=0,
            )
            fold_scores.append(score)

            trial.report(float(np.mean(fold_scores)), step=fold_number)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=max(5, n_trials // 5),
            n_warmup_steps=2,
        ),
        study_name=f"hybrid_{estimator_type}_macro_f1",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("\n" + "=" * 68)
    print("OPTUNA HYPERPARAMETER OPTIMISATION")
    print("=" * 68)
    print(f"Estimator:             {estimator_type.upper()}")
    print(f"Completed trials:      {len(study.trials)}")
    print(f"Best CV macro F1:      {study.best_value:.4f}")
    print("Best parameters:")
    for parameter, value in study.best_params.items():
        print(f"  {parameter}: {value}")
    print("=" * 68 + "\n")

    return study.best_params, study

def run_hybrid_experiment(
    architecture,
    estimator_type="rf",
    *,
    imbalance_strategy="oversample",
    scale_tabular=True,
    scale_embeddings=True,
    use_pca=False,
    pca_variance=0.95,
    checkpoint_path=None,
    use_embedding_cache=True,
    dropout=0.0,
    optimise_hyperparameters=False,
    n_trials=40,
    cv_splits=5,
):
    """
    Run one complete staged feature-fusion experiment.

    Example
    -------
    run_hybrid_experiment(
        architecture="swin_t",
        estimator_type="rf",
        imbalance_strategy="oversample",
        scale_tabular=True,
        scale_embeddings=True,
        use_pca=True,
        pca_variance=0.95,
    )
    """
    architecture = _normalise_architecture_name(architecture)
    estimator_type = str(estimator_type).strip().lower()

    train_df, test_df = load_master_splits()

    train_embeddings = extract_embeddings(
        train_df,
        architecture=architecture,
        split_name="train",
        checkpoint_path=checkpoint_path,
        use_cache=use_embedding_cache,
        dropout=dropout,
    )
    test_embeddings = extract_embeddings(
        test_df,
        architecture=architecture,
        split_name="test",
        checkpoint_path=checkpoint_path,
        use_cache=use_embedding_cache,
        dropout=dropout,
    )

    best_params = {}
    study = None
    if optimise_hyperparameters:
        best_params, study = tune_hybrid_estimator(
            train_df=train_df,
            train_embeddings=train_embeddings,
            estimator_type=estimator_type,
            imbalance_strategy=imbalance_strategy,
            scale_tabular=scale_tabular,
            scale_embeddings=scale_embeddings,
            use_pca=use_pca,
            pca_variance=pca_variance,
            n_trials=n_trials,
            n_splits=cv_splits,
        )

    prepared = prepare_fused_features(
        train_df=train_df,
        test_df=test_df,
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings,
        scale_tabular=scale_tabular,
        scale_embeddings=scale_embeddings,
        use_pca=use_pca,
        pca_variance=pca_variance,
    )

    # For RF, class_weight is another valid experiment. Do not combine it with
    # random oversampling in the primary comparison.
    rf_class_weight = None
    effective_strategy = imbalance_strategy
    if estimator_type == "rf" and imbalance_strategy in {
        "balanced",
        "balanced_subsample",
    }:
        rf_class_weight = imbalance_strategy
        effective_strategy = "none"

    balanced = balance_fused_training_data(
        prepared["X_train_fused"],
        prepared["y_train"],
        strategy=effective_strategy,
    )

    model = build_hybrid_classifier(
        estimator_type=estimator_type,
        rf_class_weight=rf_class_weight,
        model_params=best_params,
    )

    fit_kwargs = {}
    if balanced["sample_weight"] is not None:
        fit_kwargs["sample_weight"] = balanced["sample_weight"]

    model.fit(
        balanced["X_train"],
        balanced["y_train"],
        **fit_kwargs,
    )

    experiment_name = (
        f"Hybrid {architecture} + {estimator_type.upper()} "
        f"({imbalance_strategy}, PCA={use_pca})"
    )
    predictions, results = evaluate_and_save(
        experiment_name,
        model,
        prepared["X_test_fused"],
        prepared["y_test"],
    )

    safe_name = (
        f"hybrid_{architecture}_{estimator_type}_"
        f"{imbalance_strategy}_pca_{use_pca}"
    ).lower()

    artefacts = {
        "model": model,
        "architecture": architecture,
        "estimator_type": estimator_type,
        "imbalance_strategy": imbalance_strategy,
        "tabular_scaler": prepared["tabular_scaler"],
        "embedding_scaler": prepared["embedding_scaler"],
        "embedding_pca": prepared["embedding_pca"],
        "sampler": balanced["sampler"],
        "tabular_feature_names": list(FEATURES_FOR_SCORING),
        "fused_feature_names": prepared["fused_feature_names"],
        "visual_dim": prepared["visual_dim"],
        "tabular_dim": prepared["tabular_dim"],
        "image_size": IMG_SIZE,
        "class_names": CLASS_NAMES,
        "optimised_hyperparameters": optimise_hyperparameters,
        "best_params": best_params,
        "best_cv_macro_f1": (study.best_value if study is not None else None),
        "checkpoint_path": str(
            checkpoint_path
            if checkpoint_path is not None
            else ARCHITECTURE_CONFIG[architecture]["checkpoint"]
        ),
    }

    artefact_path = MODEL_DIR / f"{safe_name}_bundle.joblib"
    joblib.dump(artefacts, artefact_path)
    print("Saved complete hybrid inference bundle:", artefact_path)

    return {
        "model": model,
        "predictions": predictions,
        "results": results,
        "prepared": prepared,
        "balanced": balanced,
        "artefact_path": artefact_path,
        "best_params": best_params,
        "study": study,
    }


if __name__ == "__main__":
    # Change these values to run a different controlled experiment.
    # Recommended primary setup:
    #   - clean fine-tuned embeddings
    #   - separate StandardScaler per branch
    #   - optional PCA for the embedding branch
    #   - RandomOverSampler on the fused training rows only
    run_hybrid_experiment(
        architecture="swin_t",
        estimator_type="rf",
        imbalance_strategy="oversample",
        scale_tabular=True,
        scale_embeddings=True,
        use_pca=True,
        pca_variance=0.95,
        use_embedding_cache=True,
        optimise_hyperparameters=True,
        n_trials=40,
        cv_splits=5,
    )