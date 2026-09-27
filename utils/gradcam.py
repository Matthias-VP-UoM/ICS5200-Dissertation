from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import models
import timm
from timm.layers import resample_abs_pos_embed
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode

# Import pytorch-grad-cam classes
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM, ScoreCAM, GuidedBackpropReLUModel
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_SIZE = (320, 704)  # Image size for deep model inference

# Used to convert prediction generated as integer into a string
LABEL_MAP = {0: "High", 1: "Medium", 2: "Low"}

# Normalises any unorthodox architecture names and maintains consistency
def normalise_architecture_name(architecture: str) -> str:
    aliases = {
        "rn18": "resnet18",
        "resnet_18": "resnet18",
        "rn50": "resnet50",
        "resnet_50": "resnet50",
        "vgg": "vgg16",
        "vit": "vit_b_16",
        "vit_b16": "vit_b_16",
        "vit-b-16": "vit_b_16",
        "vit-b/16": "vit_b_16",
        "vit_b/16": "vit_b_16",
        "swin": "swin_t",
        "swin-t": "swin_t",
    }
    architecture = str(architecture).strip().lower()
    return aliases.get(architecture, architecture)

# Defines the dropout head layer for each deep learning model
def checkpoint_uses_dropout_head(state_dict, architecture):
    sequential_weight_keys = {
        "resnet18": "fc.1.weight",
        "resnet50": "fc.1.weight",
        "vgg16": "classifier.6.1.weight",
        "vit_b_16": "heads.head.1.weight",
        "swin_t": "head.1.weight",
    }
    return sequential_weight_keys.get(architecture) in state_dict

# Function which is used to build each of the deep learning models
def build_model(architecture: str, num_classes: int = 3, force_sequential_head: bool = False):
    architecture = normalise_architecture_name(architecture)
    if architecture == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = (
            nn.Sequential(nn.Dropout(p=0.0), nn.Linear(in_features, num_classes))
            if force_sequential_head else nn.Linear(in_features, num_classes)
        )
        target_layers = [model.layer4[-1]]
    elif architecture == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = (
            nn.Sequential(nn.Dropout(p=0.0), nn.Linear(in_features, num_classes))
            if force_sequential_head else nn.Linear(in_features, num_classes)
        )
        target_layers = [model.layer4[-1]]
    elif architecture == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        in_features = model.classifier[6].in_features
        model.classifier[6] = (
            nn.Sequential(nn.Dropout(p=0.0), nn.Linear(in_features, num_classes))
            if force_sequential_head else nn.Linear(in_features, num_classes)
        )
        target_layers = [model.features[-1]]
    elif architecture == "vit_b_16":
        model = timm.create_model('vit_base_patch16_224', pretrained=False, dynamic_img_size=True, img_size=(320, 704))
        in_features = model.head.in_features
        model.head = (
            nn.Sequential(nn.Dropout(p=0.0), nn.Linear(in_features, num_classes))
            if force_sequential_head else nn.Linear(in_features, num_classes)
        )
        target_layers = [model.blocks[-1].norm1]
    elif architecture in {"swin_t", "swin-t"}:
        model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        model.head = nn.Linear(model.head.in_features, num_classes)
        target_layers = [model.features[-1]]
    else:
        raise ValueError(f"Unsupported Grad-CAM architecture: {architecture}")
    return model, target_layers

# Child version of the build_model() function which is adapted for the LAYERCAM approach
def build_model_layercam(architecture: str, num_classes: int = 3, force_sequential_head: bool = False):
    architecture = (architecture)
    if architecture == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = (
            nn.Sequential(nn.Dropout(p=0.0), nn.Linear(in_features, num_classes))
            if force_sequential_head else nn.Linear(in_features, num_classes)
        )
        # Combine intermediate + deep features to recover fine-grained resolution
        target_layers = [model.layer3[-1], model.layer4[-1]]
    elif architecture == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = (
            nn.Sequential(nn.Dropout(p=0.0), nn.Linear(in_features, num_classes))
            if force_sequential_head else nn.Linear(in_features, num_classes)
        )
        target_layers = [model.layer3[-1], model.layer4[-1]]
    elif architecture == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        in_features = model.classifier[6].in_features
        model.classifier[6] = (
            nn.Sequential(nn.Dropout(p=0.0), nn.Linear(in_features, num_classes))
            if force_sequential_head else nn.Linear(in_features, num_classes)
        )
        target_layers = [model.features[21], model.features[28]]
    elif architecture == "vit_b_16":
        model = timm.create_model('vit_base_patch16_224', pretrained=False, dynamic_img_size=True, img_size=(320, 704))
        in_features = model.head.in_features
        model.head = (
            nn.Sequential(nn.Dropout(p=0.0), nn.Linear(in_features, num_classes))
            if force_sequential_head else nn.Linear(in_features, num_classes)
        )
        target_layers = [model.blocks[-2].norm1, model.blocks[-1].norm1]
    elif architecture in {"swin_t", "swin-t"}:
        model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        model.head = nn.Linear(model.head.in_features, num_classes)
        target_layers = [model.features[5], model.features[7]]
    else:
        raise ValueError(f"Unsupported LayerCAM architecture: {architecture}")
    return model, target_layers

# Loads the deep learning model for inference
def load_visual_model(checkpoint_path: str | Path, architecture: str, device: torch.device, is_layercam: bool = False):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Visual checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint.get("model", checkpoint)))
    else:
        state_dict = checkpoint

    state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    norm_architecture = (architecture)
    uses_dropout_head = (state_dict, norm_architecture)

    if norm_architecture == "vit_b_16" and "pos_embed" in state_dict:
        checkpoint_pos_embed = state_dict["pos_embed"]
        target_num_tokens = 20 * 44 + 1  # 881, matches 320x704 input grid

        if checkpoint_pos_embed.shape[1] != target_num_tokens:
            old_num_patches = checkpoint_pos_embed.shape[1] - 1
            old_grid_size = int(round(old_num_patches ** 0.5))
            state_dict["pos_embed"] = resample_abs_pos_embed(
                checkpoint_pos_embed,
                new_size=[20, 44],
                old_size=[old_grid_size, old_grid_size],
                num_prefix_tokens=1
            )

    # Checks if LAYERCAM is being used - if yes, use build_model_layercam(); if no, use the standard build_model()
    builder_fn = build_model_layercam if is_layercam else build_model
    model, target_layers = builder_fn(norm_architecture, force_sequential_head=uses_dropout_head)
    
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model, target_layers

# Preprocesses the webpage screenshot input before it is passed to the selected GRAD-CAM technique
def preprocess(image: Image.Image, image_size=DEFAULT_IMAGE_SIZE):
    transform = v2.Compose(
        [
            v2.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0)


def vit_reshape_transform(tensor):
    if tensor.ndim == 3:
        if tensor.size(1) == 881:
            spatial_tokens = tensor[:, 1:, :]
        else:
            spatial_tokens = tensor
            
        batch_size, num_tokens, channels = spatial_tokens.shape
        if num_tokens == 880:  # 20 * 44
            result = spatial_tokens.reshape(batch_size, 20, 44, channels)
            return result.permute(0, 3, 1, 2)
            
    return tensor


def swin_reshape_transform(tensor):
    if tensor.ndim == 4:
        return tensor.permute(0, 3, 1, 2)
    elif tensor.ndim == 3:
        batch_size, num_tokens, channels = tensor.shape
        h = w = int(np.sqrt(num_tokens))
        return tensor.permute(0, 2, 1).reshape(batch_size, channels, h, w)
    return tensor

# Initialises the Grad-CAM method based on what was selected by the user
def get_cam_method(method_name: str, model: nn.Module, target_layers: list[nn.Module], architecture: str):
    norm_arch = (architecture)

    if norm_arch == "vit_b_16":
        reshape_transform = vit_reshape_transform
    elif norm_arch == "swin_t":
        reshape_transform = swin_reshape_transform
    else:
        reshape_transform = None

    method_name = method_name.lower().replace("-", "").replace("_", "")
    
    if method_name == "gradcam":
        return GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    elif method_name in {"gradcam++", "gradcampp"}:
        return GradCAMPlusPlus(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    elif method_name == "layercam":
        return LayerCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    elif method_name == "scorecam":
        return ScoreCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    else:
        raise ValueError(f"Unsupported method: {method_name}. Options: 'gradcam', 'gradcam++', 'layercam', 'scorecam', 'guided-gradcam'")

# Computes Drop-in-Confidence and Increase-in-Confidence quantitative metrics
def calculate_cam_metrics(model: nn.Module, input_tensor: torch.Tensor, grayscale_cam: np.ndarray, class_index: int) -> dict[str, float]:
    with torch.no_grad():
        orig_logits = model(input_tensor)
        orig_conf = float(orig_logits.softmax(dim=1)[0, class_index].cpu().item())

        cam_mask = torch.from_numpy(grayscale_cam).unsqueeze(0).unsqueeze(0).to(input_tensor.device)
        cam_mask = torch.nn.functional.interpolate(cam_mask, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)

        # Masked input keeping only highlighted features
        masked_input = input_tensor * cam_mask
        masked_logits = model(masked_input)
        masked_conf = float(masked_logits.softmax(dim=1)[0, class_index].cpu().item())

    drop_in_confidence = max(0.0, (orig_conf - masked_conf) / (orig_conf + 1e-8))
    increase_in_confidence = 1.0 if masked_conf > orig_conf else 0.0

    return {
        "drop_in_confidence": drop_in_confidence,
        "increase_in_confidence": increase_in_confidence,
        "original_confidence": orig_conf,
        "masked_confidence": masked_conf,
    }

# Generates the Grad-CAM overlay using the selected technique
def create_gradcam_overlay(screenshot_path: str, checkpoint_path: str | Path, architecture: str, method: str = "layercam", class_index: int | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(screenshot_path).convert("RGB")
    input_tensor = preprocess(image).to(device)

    clean_method = method.lower().replace("-", "").replace("_", "")
    is_layercam = clean_method == "layercam"

    model, target_layers = load_visual_model(checkpoint_path, architecture, device, is_layercam=is_layercam)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = logits.softmax(dim=1)[0].cpu().numpy()
        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())

    targets = [ClassifierOutputTarget(class_index)]
    norm_arch = (architecture)

    if clean_method == "guidedgradcam":
        reshape_transform = (
            vit_reshape_transform if norm_arch == "vit_b_16"
            else swin_reshape_transform if norm_arch == "swin_t"
            else None
        )
        cam_generator = GradCAM(
            model=model,
            target_layers=target_layers,
            reshape_transform=reshape_transform
        )
        grayscale_cam = cam_generator(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=True)[0]

        gb_model = GuidedBackpropReLUModel(model=model, device=device)
        gb_map = gb_model(input_tensor, target_category=class_index)

        if gb_map.ndim == 3 and gb_map.shape[0] == 3:
            gb_map = np.transpose(gb_map, (1, 2, 0))

        # 2. Colorize the Grad-CAM heatmap and blend with original image
        original = np.asarray(image.resize((DEFAULT_IMAGE_SIZE[1], DEFAULT_IMAGE_SIZE[0])))
        heatmap_uint8 = np.uint8(255 * grayscale_cam)
        coloured = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)

        # Blend original screenshot with the Grad-CAM heatmap
        overlay = np.clip(0.58 * original + 0.42 * coloured, 0, 255).astype(np.uint8)
        
        # heatmap = gb_map * grayscale_cam[..., np.newaxis]
        # heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # original = np.asarray(image.resize((DEFAULT_IMAGE_SIZE[1], DEFAULT_IMAGE_SIZE[0])))
        # overlay = np.clip(heatmap * 255, 0, 255).astype(np.uint8)

    else:
        cam_generator = get_cam_method(method, model, target_layers, architecture)
        grayscale_cam = cam_generator(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=True)[0]

        original = np.asarray(image.resize((DEFAULT_IMAGE_SIZE[1], DEFAULT_IMAGE_SIZE[0])))
        heatmap_uint8 = np.uint8(255 * grayscale_cam)
        coloured = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)
        overlay = np.clip(0.58 * original + 0.42 * coloured, 0, 255).astype(np.uint8)
    
    cam_metrics = calculate_cam_metrics(model, input_tensor, grayscale_cam, class_index)

    overlay_img = Image.fromarray(overlay)
    width, height = overlay_img.size
    banner_height = 40
    
    final_img = Image.new("RGB", (width, height + banner_height), color=(255, 255, 255))
    final_img.paste(overlay_img, (0, banner_height))
    
    pred_label = LABEL_MAP.get(class_index, f"Class {class_index}")
    confidence = probabilities[class_index]
    header_text = (
        f"Pred: {pred_label} ({confidence:.2%}) | Method: {method.upper()} | "
        f"Drop-in-Conf: {cam_metrics['drop_in_confidence']:.1%}"
    )

    draw = ImageDraw.Draw(final_img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    draw.text((15, 10), header_text, fill=(0, 0, 0), font=font)

    return final_img, probabilities, class_index, cam_metrics