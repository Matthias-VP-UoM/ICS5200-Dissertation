import matplotlib.pyplot as plt
import optuna
from optuna.trial import TrialState
import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import timm
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from models.early_stopping import EarlyStopping
from utils.image_loader import WebpageScreenshotDataset, get_image_transforms, create_weighted_sampler, IMG_SIZE

def build_vit(num_classes, dropout=0.0, drop_path=0.0, attn_drop=0.0):
    model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=True,
        img_size=IMG_SIZE,
        drop_rate=dropout, # Connects the model's inner dropout rate to Optuna's parameter
        drop_path_rate=drop_path,
        attn_drop_rate=attn_drop,
        num_classes=num_classes
    )

    return model

def build_swin_t(num_classes, dropout=0.0):
    """
    Swin Transformer Tiny backbone. Unlike plain ViT, Swin has a 4-stage
    hierarchical structure with local windowed attention, producing genuine
    spatial feature maps at multiple resolutions - this is what makes it
    Grad-CAM friendly (hook the last stage's feature map exactly like a CNN
    conv layer) instead of requiring attention-rollout.
    """
    weights = models.Swin_T_Weights.DEFAULT
    model = models.swin_t(weights=weights)

    in_features = model.head.in_features
    if dropout > 0.0:
        model.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
    else:
        model.head = nn.Linear(in_features, num_classes)

    return model


def vit_init_transform(image_size=IMG_SIZE):
    train_transform = get_image_transforms(image_size, is_train=True)
    test_transform = get_image_transforms(image_size, is_train=False)

    return train_transform, test_transform


def vit_init_dataset(train_df, test_df, train_transform, test_transform):
    train_dataset = WebpageScreenshotDataset(train_df, transform=train_transform)
    test_dataset = WebpageScreenshotDataset(test_df, transform=test_transform)

    return train_dataset, test_dataset


def vit_init_loader(train_dataset, test_dataset, batch_size=16, use_weighted_sampler=True):
    colab_workers = os.cpu_count()

    if use_weighted_sampler:
        train_sampler = create_weighted_sampler(
            train_dataset
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False, num_workers=colab_workers, pin_memory=torch.cuda.is_available(), persistent_workers=colab_workers > 0)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=colab_workers, pin_memory=torch.cuda.is_available(), persistent_workers=colab_workers > 0)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=colab_workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=colab_workers, pin_memory=True, persistent_workers=True)

    return train_loader, test_loader


def tune_vit(train_dt, test_dt, num_classes=3, n_trials=5):
    """
    Optuna study wrapper to optimize hyper-parameters for Swin-T training.
    Swin-T (~28M params) is lighter than ViT-B/16 (~86M params), so larger
    batch sizes are generally feasible even at the bigger IMG_SIZE.
    """

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-6, 5e-4, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16])
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.3, step=0.1)

        max_epochs = 5

        train_loader, test_loader = vit_init_loader(train_dt, test_dt, batch_size=batch_size)

        print(f"\n--- Starting Trial {trial.number} | lr: {lr:.6f}, batch_size: {batch_size}, weight_decay: {weight_decay}, dropout: {dropout} ---")

        try:
            unique_checkpoint = f"vit_trial_{trial.number}"

            _, history, _ = train_vit_model(
                train_loader=train_loader,
                test_loader=test_loader,
                num_classes=num_classes,
                epochs=max_epochs,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                dropout=dropout,
                checkpoint_name=unique_checkpoint
            )

            best_val_acc = max(history["test_acc"])
            return best_val_acc

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"Trial {trial.number} failed due to CUDA OOM. Skipping...")
                torch.cuda.empty_cache()
                return 0.0
            else:
                raise e

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler())
    study.optimize(objective, n_trials=n_trials)

    print("\nOptimization Complete!")
    print("Best Parameters:", study.best_params)
    print(f"Best Test Accuracy: {study.best_value:.4f}")

    return study.best_params


def train_vit_model(train_loader, test_loader, num_classes=3, epochs=20, batch_size=16, lr=1e-5, weight_decay=1e-4, dropout=0.0, checkpoint_name=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_swin_t(num_classes, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    # Swin/ViT-style transformers generally prefer AdamW due to weight decay requirements
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu")

    if checkpoint_name:
        checkpoint_file = f"outputs/models/{checkpoint_name}.pth"
    else:
        checkpoint_file = "outputs/models/swin_t_best_.pth"
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    early_stopping = EarlyStopping(patience=3, checkpoint_path=checkpoint_file)

    history = {
        "train_loss": [],
        "test_loss": [],
        "test_acc": []
    }

    best_checkpoint_acc = 0.0

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimiser.zero_grad()

            with torch.amp.autocast(device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # --- Evaluation Phase ---
        model.eval()
        total_test_loss = 0
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                with torch.amp.autocast(device_type=device.type):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                total_test_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        avg_test_loss = total_test_loss / len(test_loader)
        accuracy = correct / total

        history["train_loss"].append(avg_train_loss)
        history["test_loss"].append(avg_test_loss)
        history["test_acc"].append(accuracy)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Accuracy: {accuracy:.4f}")

        early_stopping(avg_test_loss, model)

        if early_stopping.best_loss == avg_test_loss:
            best_checkpoint_acc = accuracy

        if early_stopping.early_stop:
            print(f"Early stopping triggered! Training stopped at epoch {epoch+1}.")
            break

    print("Loading best Swin-T weights from checkpoint...")
    model.load_state_dict(torch.load(checkpoint_file))

    if checkpoint_name and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    return model, history, best_checkpoint_acc


def plot_vit_loss_graph(history):
    epochs_range = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss", color="blue", marker="o")
    plt.plot(epochs_range, history["test_loss"], label="Test Loss", color="red", marker="o")
    plt.title("Training and Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["test_acc"], label="Test Accuracy", color="green", marker="s")
    plt.title("Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("outputs/plots/swin_t_training_metrics.png")
    plt.show()


def evaluate_vit_performance(model, test_df, batch_size=16, class_names=None):
    """
    High-level wrapper that manages its own data loading and handles the
    evaluation pipeline cleanly.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    test_transform = get_image_transforms(IMG_SIZE, is_train=False)
    test_dataset = WebpageScreenshotDataset(test_df, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return retrieve_eval_metrics_vit(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names
    )


def retrieve_eval_metrics_vit(model, test_loader, device, class_names=None):
    """
    Evaluates a PyTorch model using Accuracy, Precision, Recall, F1-Score,
    and plots a Confusion Matrix.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)

            predictions = torch.argmax(outputs, dim=1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print("\n=== Evaluation Metrics ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("==========================\n")

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("Classification Report:\n", report)

    plt.figure(figsize=(8, 6))

    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(cm))]

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    plt.savefig("outputs/plots/swin_t_confusion_matrix.png")
    plt.show()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }
