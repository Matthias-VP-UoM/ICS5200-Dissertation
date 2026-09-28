import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode


class WebpageScreenshotDataset(Dataset):
    def __init__(self, df, image_col="screenshot_path", label_col="accessibility_label", transform=None):
        self.df = df.reset_index(drop=True)
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.loc[idx, self.image_col]
        label = int(self.df.loc[idx, self.label_col])

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

IMG_SIZE = (320, 704)

# Perform transformations on the images to prep them for use in models (both training and inference)
def get_image_transforms(image_size=IMG_SIZE, is_train=False):
    structural_transforms = [
        v2.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
    ]

    normalization_transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
    if is_train:
        return v2.Compose([
            *structural_transforms,
            # Simulate different screen brightness/contrast profiles
            v2.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.0,              # Keep colours intact if color matters for your classes
                hue=0.0
            ),
            *normalization_transforms
        ])
    else:
        # Testing pipeline remains strictly clean and deterministic
        return v2.Compose([
            *structural_transforms,
            *normalization_transforms
        ])

# Creates a WeightedRandomSampler from the labels stored in WebpageScreenshotDataset
# Minority-class samples receive a higher probability of being drawn
def create_weighted_sampler(dataset,num_samples=None,replacement=True):
    labels = (
        dataset.df[dataset.label_col]
        .astype(int)
        .to_numpy()
    )

    unique_classes, class_counts = np.unique(
        labels,
        return_counts=True
    )

    class_weights = {
        class_label: len(labels) / (
            len(unique_classes) * class_count
        )
        for class_label, class_count
        in zip(unique_classes, class_counts)
    }

    sample_weights = np.array(
        [class_weights[label] for label in labels],
        dtype=np.float64
    )

    if num_samples is None:
        num_samples = len(labels)

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(
            sample_weights,
            dtype=torch.double
        ),
        num_samples=num_samples,
        replacement=replacement
    )

    print("Original class counts:")
    for class_label, class_count in zip(
        unique_classes,
        class_counts
    ):
        print(
            f"  Class {class_label}: "
            f"{class_count} samples, "
            f"weight={class_weights[class_label]:.4f}"
        )

    return sampler
