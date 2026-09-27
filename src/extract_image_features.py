import os
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import shutil

# Function used to calculate edge density within a given screenshot
def compute_edge_density(gray):
    edges = cv2.Canny(gray, 100, 200)
    return np.sum(edges > 0) / edges.size

# Function used to calculate contrast within a given screenshot
def compute_contrast(gray):
    return float(np.std(gray))

# Function used to calculate colour variance within a given screenshot
def compute_colour_variance(image):
    return float(np.var(image))

# Function used to calculate layout density within a given screenshot
def compute_layout_density(gray):
    threshold = 245
    non_background = gray < threshold
    return np.sum(non_background) / gray.size

# Function used to calculate text density proxy within a given screenshot
def compute_text_density_proxy(gray):
    edges = cv2.Canny(gray, 50, 150)
    return np.sum(edges > 0) / edges.size

# Function used to calculate whitespace ratio within a given screenshot
def compute_whitespace_ratio(gray):
    threshold = 245
    whitespace = gray >= threshold
    return np.sum(whitespace) / gray.size

# Main function of this script - extracts the above features and saves them in a CSV file
def extract_image_features(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.resize(image, (512, 512))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features = {
        "edge_density": compute_edge_density(gray),
        "contrast": compute_contrast(gray),
        "colour_variance": compute_colour_variance(image),
        # "layout_density": compute_layout_density(gray),
        "text_density_proxy": compute_text_density_proxy(gray),
        "mean_brightness": float(np.mean(gray)),
        "brightness_variance": float(np.var(gray)),
        "whitespace_ratio": compute_whitespace_ratio(gray),
    }

    return features


def main():
    manifest_path = "manifest.csv"
    output_dir = "data/features/image"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # else:
    #     dir_path = Path(output_dir)

    #     for item in dir_path.iterdir():
    #         if item.is_dir():
    #             shutil.rmtree(item)
    #         else:
    #             item.unlink()

    #     exit(0)

    manifest = pd.read_csv(manifest_path)
    # tom_manifest = manifest[manifest['website'] == 'times_of_malta']

    all_features = []

    # tqdm(imgs_list_full, desc='Processing images...')
    for _, row in tqdm(manifest.iterrows(), desc='Processing images...'):
        page_id = str(row["id"]).zfill(4)
        image_path = row["screenshot_path"]

        if not os.path.exists(image_path):
            print(f"Missing screenshot: {image_path}")
            continue

        features = extract_image_features(image_path)
        features["id"] = page_id
        features["website"] = row["website"]
        features["page_type"] = row["page_type"]

        # Save to JSON
        site_dir = os.path.join(output_dir, row["website"])
        if not os.path.exists(site_dir):
            os.makedirs(site_dir)

        json_path = os.path.join(site_dir, f"{page_id}.json")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(features, f, indent=4)

        all_features.append(features)

    df = pd.DataFrame(all_features)
    df.to_csv("data/features/image_features_v2.csv", index=False)

    print("Image feature extraction complete.")


if __name__ == "__main__":
    main()