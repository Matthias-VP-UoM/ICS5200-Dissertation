import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from models.early_stopping import EarlyStopping
from utils.image_loader import WebpageScreenshotDataset, get_image_transforms, create_weighted_sampler, IMG_SIZE

# class EarlyStopping:
#     def __init__(self, patience=3, min_delta=0, checkpoint_path="outputs/models/cnn_resnet18.pth"):
#         """
#         Args:
#             patience (int): How many epochs to wait after last time validation loss improved.
#             min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
#             checkpoint_path (str): Path to save the best model weights.
#         """
#         self.patience = patience
#         self.min_delta = min_delta
#         self.checkpoint_path = checkpoint_path
#         self.counter = 0
#         self.best_loss = None
#         self.early_stop = False

#     def __call__(self, val_loss, model):
#         if self.best_loss is None:
#             self.best_loss = val_loss
#             self.save_checkpoint(model)
#         elif val_loss > self.best_loss - self.min_delta:
#             self.counter += 1
#             print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_loss = val_loss
#             self.save_checkpoint(model)
#             self.counter = 0  # Reset counter since we found a better model

#     def save_checkpoint(self, model):
#         """Saves model when validation loss decreases."""
#         torch.save(model.state_dict(), self.checkpoint_path)
#         print(f"--> Validation loss decreased. Saving best model model configuration...")

def build_resnet18(num_classes, dropout=0.0):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    if dropout > 0.0:
        # Reconstruct the fc layer to include Dropout
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
    else:
        model.fc = nn.Linear(in_features, num_classes)
    # model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_resnet50(num_classes, dropout=0.0):
    """
    ResNet50 backbone. Swapped from VGG16/ResNet18.
    For Grad-CAM mapping later on, model.layer4[-1] serves as the
    ideal target feature map layer.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features

    if dropout > 0.0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
    else:
        model.fc = nn.Linear(in_features, num_classes)
    return model


def build_vgg16(num_classes, dropout=0.0):
    """
    VGG16 backbone. `model.features` is a plain conv stack (no skip
    connections), which is the reason it is preferred over ResNet for
    Grad-CAM style explainability - gradients cannot "skip around" the
    probed layer, so activation maps localize more cleanly. The final conv
    layer of `model.features` (before AdaptiveAvgPool2d) is the natural
    Grad-CAM target layer.
    """
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    in_features = model.classifier[6].in_features
    if dropout > 0.0:
        model.classifier[6] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
    else:
        model.classifier[6] = nn.Linear(in_features, num_classes)
    return model


def cnn_init_transform(image_size=IMG_SIZE):
    train_transform = get_image_transforms(image_size, is_train=True)
    test_transform = get_image_transforms(image_size, is_train=False)

    return train_transform, test_transform

def cnn_init_dataset(train_df, test_df, train_transform, test_transform):
    train_dataset = WebpageScreenshotDataset(train_df, transform=train_transform)
    test_dataset = WebpageScreenshotDataset(test_df, transform=test_transform)

    return train_dataset, test_dataset

def cnn_init_loader(train_dataset, test_dataset, batch_size=16, use_weighted_sampler=True):
    colab_workers = os.cpu_count()

    if use_weighted_sampler:
        train_sampler = create_weighted_sampler(
            train_dataset
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False, num_workers=colab_workers, pin_memory=torch.cuda.is_available(), persistent_workers=colab_workers > 0)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=colab_workers, pin_memory=torch.cuda.is_available(), persistent_workers=colab_workers > 0)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=colab_workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=colab_workers, pin_memory=torch.cuda.is_available(), persistent_workers=True)

    return train_loader, test_loader

# --- NEW: Added Optuna Optimization Wrapper ---
def tune_cnn(train_dt, test_dt, num_classes, n_trials=5):
    """
    Optuna study wrapper to optimize hyperparameters for CNN (ResNet18)
    training.
    """

    def objective(trial):
        # ResNet models typically tolerate slightly larger learning rates than ViTs
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        # batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        batch_size = trial.suggest_categorical("batch_size", [8, 16])
        optimizer = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)

        max_epochs = 5  # High ceiling, early stopping handles the rest

        train_loader, test_loader = cnn_init_loader(train_dt, test_dt, batch_size=batch_size)

        print(f"\n--- Starting CNN Trial {trial.number} | lr: {lr:.6f}, batch_size: {batch_size}, optimizer: {optimizer}, weight_decay: {weight_decay}, dropout: {dropout} ---")

        try:
            unique_checkpoint = f"cnn_trial_{trial.number}"

            # We call the training loop using a temporary checkpoint path
            _, history, _ = train_cnn_model(
                train_loader=train_loader,
                test_loader=test_loader,
                num_classes=num_classes,
                epochs=max_epochs,
                batch_size=batch_size,
                lr=lr,
                optimizer=optimizer,
                weight_decay=weight_decay,
                dropout=dropout,
                checkpoint_name=unique_checkpoint,
                trial=trial
            )
            return max(history["test_acc"])
        except optuna.exceptions.TrialPruned:
            # If the inner function raised a Pruned exception, pass it up to Optuna
            raise
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"Trial {trial.number} failed due to CUDA OOM. Skipping...")
                torch.cuda.empty_cache()
                return 0.0
            raise e

        del _, history  # or whatever model variables are hanging around
        torch.cuda.empty_cache()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler())
    study.optimize(objective, n_trials=n_trials)

    # Get all trials
    trials = study.trials

    completed_trials = [t for t in trials if t.state == TrialState.COMPLETE]
    pruned_trials = [t for t in trials if t.state == TrialState.PRUNED]
    failed_trials = [t for t in trials if t.state == TrialState.FAIL]

    print("--- Optuna Study Summary ---")
    print(f"Total Trials: {len(trials)}")
    print(f"Successfully Completed Trials: {len(completed_trials)}")
    print(f"Pruned (Stopped Early) Trials: {len(pruned_trials)}")
    print(f"Failed (Crashed) Trials: {len(failed_trials)}")

    if study.best_trial:
        print("\n--- Best Trial Results ---")
        print(f"  Trial Number: {study.best_trial.number}")
        print(f"  Best Accuracy: {study.best_trial.value:.4f}")
        print("  Best Hyperparameters:")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")

    # print("\n=== CNN Optimization Complete ===")
    # print("Best CNN Parameters Found:", study.best_params)
    return study.best_params

# def train_cnn_model(train_loader, test_loader, num_classes=3, epochs=20, batch_size=16, lr=1e-4, optimizer="Adam", weight_decay=0.001, dropout=0.0, checkpoint_name=None):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model = build_resnet18(num_classes, dropout=dropout).to(device)
#     criterion = nn.CrossEntropyLoss()
#     # optimiser = torch.optim.Adam(model.parameters(), lr=lr)

#     # 3. Dynamically choose the Optimizer and apply Weight Decay
#     if optimizer == "AdamW":
#         optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#     elif optimizer == "SGD":
#         # Adding a standard momentum of 0.9 for SGD
#         optimiser = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
#     else:
#         optimiser = torch.optim.Adam(model.parameters(), lr=lr)

#     # --- 1. Initialize Early Stopping Helper ---
#     if checkpoint_name:
#         checkpoint_file = f"outputs/models/{checkpoint_name}.pth"
#     else:
#         checkpoint_file = "outputs/models/cnn_resnet18_best_.pth"

#     # Ensure the directory exists
#     os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
#     early_stopping = EarlyStopping(patience=3, checkpoint_path=checkpoint_file)

#     history = {
#         "train_loss": [],
#         "test_loss": [],
#         "test_acc": []
#     }

#     # Tracking variable to return the exact accuracy of our best checkpoint
#     best_checkpoint_acc = 0.0

#     for epoch in range(epochs):
#         # Training Phase
#         model.train()
#         total_train_loss = 0
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)

#             optimiser.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimiser.step()
#             total_train_loss += loss.item()

#         avg_train_loss = total_train_loss / len(train_loader)

#         # Evaluation Phase
#         model.eval()
#         total_test_loss = 0
#         correct, total = 0, 0

#         with torch.no_grad():
#             for images, labels in test_loader:
#                 images, labels = images.to(device), labels.to(device)
#                 outputs = model(images)

#                 loss = criterion(outputs, labels)
#                 total_test_loss += loss.item()

#                 predictions = torch.argmax(outputs, dim=1)
#                 total += labels.size(0)
#                 correct += (predictions == labels).sum().item()

#         avg_test_loss = total_test_loss / len(test_loader)
#         accuracy = correct / total

#         history["train_loss"].append(avg_train_loss)
#         history["test_loss"].append(avg_test_loss)
#         history["test_acc"].append(accuracy)

#         print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Accuracy: {accuracy:.4f}")

#         # --- 2. Check Early Stopping Status ---
#         # If this epoch has the lowest loss, early_stopping updates internal state.
#         # We capture the accuracy *of this specific best epoch*.
#         # is_baseline = early_stopping.best_loss is None
#         # is_improvement = not is_baseline and (avg_test_loss <= early_stopping.best_loss - early_stopping.min_delta)

#         # if is_baseline or is_improvement:
#         #     best_checkpoint_acc = accuracy
#         # Pass the current epoch's validation loss and the model
#         early_stopping(avg_test_loss, model)

#         if early_stopping.best_loss == avg_test_loss:
#             best_checkpoint_acc = accuracy

#         if early_stopping.early_stop:
#             print(f"Early stopping triggered! Training stopped at epoch {epoch+1}.")
#             break

#     # --- 3. Load the absolute best weights before returning ---
#     print("Loading best model weights from checkpoint...")
#     model.load_state_dict(torch.load(checkpoint_file))

#     if checkpoint_name and os.path.exists(checkpoint_file):
#         os.remove(checkpoint_file)

#     return model, history, best_checkpoint_acc


def train_cnn_model(train_loader, test_loader, num_classes=3, epochs=20, batch_size=16, lr=1e-4, optimizer="Adam", weight_decay=0.001, dropout=0.0, checkpoint_name=None, trial=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = build_resnet18(num_classes, dropout=dropout).to(device)
    # model = build_resnet50(num_classes, dropout=dropout).to(device)
    model = build_vgg16(num_classes, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer == "AdamW":
        optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "SGD":
        optimiser = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    # NEW: Initialize Gradient Scaler for Mixed Precision
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu")

    if checkpoint_name:
        checkpoint_file = f"outputs/models/{checkpoint_name}.pth"
    else:
        checkpoint_file = "outputs/models/cnn_resnet18_best_.pth"

    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    early_stopping = EarlyStopping(patience=3, checkpoint_path=checkpoint_file)

    history = {"train_loss": [], "test_loss": [], "test_acc": []}
    best_checkpoint_acc = 0.0

    for epoch in range(epochs):
        # Training Phase
        model.train()
        total_train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimiser.zero_grad()

            # NEW: Wrap forward pass in autocast
            with torch.amp.autocast(device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # NEW: Scale loss, perform backward pass, and update weights using scaler
            scaler.scale(loss).backward() # Correctly scale and then call backward
            scaler.step(optimiser)        # Unscale gradients and apply optimizer step
            scaler.update()               # Update the scaler for the next iteration

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Evaluation Phase (Keep torch.no_grad(), autocast is optional here but clean)
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

        if trial is not None:
            # Report intermediate result to Optuna
            trial.report(accuracy, epoch)

            # Check if Optuna thinks this trial should be terminated early
            if trial.should_prune():
                print(f"Trial {trial.number} pruned early at epoch {epoch} due to poor performance.")
                # Force clean up before raising the exception
                del model, optimizer
                gc.collect()
                torch.cuda.empty_cache()

                # Raise the explicit exception Optuna uses to catch a pruned trial
                raise optuna.exceptions.TrialPruned()

    print("Loading best model weights from checkpoint...")
    model.load_state_dict(torch.load(checkpoint_file))

    if checkpoint_name and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    return model, history, best_checkpoint_acc


def plot_cnn_loss_graph(history):
    epochs_range = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # Plot 1: Training & Test Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss", color="blue", marker="o")
    plt.plot(epochs_range, history["test_loss"], label="Test Loss", color="red", marker="o")
    plt.title("Training and Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot 2: Test Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["test_acc"], label="Test Accuracy", color="green", marker="s")
    plt.title("Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Optional: Save the plot as an image
    plt.savefig("outputs/plots/cnn_training_metrics_.png")
    plt.show()

def evaluate_cnn_performance(model, test_df, batch_size=16, class_names=None):
    """
    High-level wrapper inside CNN.py that manages its own data loading
    and handles the evaluation pipeline cleanly.
    """
    # 1. Deduce the system hardware configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 2. Build the dataset and dataloader internally
    test_transform = get_image_transforms(IMG_SIZE, is_train=False)
    test_dataset = WebpageScreenshotDataset(test_df, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 3. Hand over execution context to your evaluation engine
    return retrieve_eval_metrics_cnn(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names
    )

def retrieve_eval_metrics_cnn(model, test_loader, device, class_names=None):
    """
    Evaluates a PyTorch model using Accuracy, Precision, Recall, F1-Score,
    and plots a Confusion Matrix.
    """
    model.eval()
    all_preds = []
    all_labels = []

    # 1. Gather all predictions and true labels
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)

            predictions = torch.argmax(outputs, dim=1)

            # Move to CPU and convert to numpy for Scikit-Learn
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 2. Calculate Classification Metrics
    accuracy = accuracy_score(all_labels, all_preds)

    # 'macro' calculates metrics for each class and finds their unweighted mean.
    # Use 'weighted' instead if you have severe class imbalance and want to account for it.
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print("\n=== Evaluation Metrics ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("==========================\n")

    # 3. Generate and Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("Classification Report:\n", report)

    plt.figure(figsize=(8, 6))

    # If class names aren't provided, use numbers
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

    # Save before showing to avoid the blank canvas bug
    plt.savefig("outputs/plots/cnn_confusion_matrix.png")
    plt.show()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }