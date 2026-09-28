import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


# Initialising paths and directories
INPUT_PATH = "data/final_dataset.csv"
OUTPUT_PATH = "data/annotated_dataset.csv"

# Configuring preprocessing options
NORMALISATION_METHOD = "standard"  

ARIA_FEATURES = [
    "has_aria_attributes",
    "num_aria_attributes",
    "aria_attribute_density",
    "num_aria_label",
    "num_aria_labelledby",
    "num_aria_hidden",
    "num_aria_live",
    "num_broken_aria_labelledby_refs",
    "interactive_elements_without_label"
]

# Define features that to be used to calculate the accessibility score for each webpage screenshot
FEATURES_FOR_SCORING = [
    # Existing screenshot-based visual features (6)
    "edge_density",
    "contrast",
    "colour_variance",
    # "layout_density",
    "text_density_proxy",
    "brightness_variance",

    # Existing DOM and content features (8)
    "dom_depth",
    "num_links",
    "num_images",
    "num_buttons",
    "num_forms",
    "num_paragraphs",
    "num_headings",
    "word_count",

    # Existing advertisement / embedded-content features (2)
    "num_advertisement_iframes",
    "num_non_ad_iframes",

    # Existing ARIA and accessible-naming features (7)
    "num_aria_attributes",
    "aria_attribute_density",
    "num_aria_label",
    "num_aria_labelledby",
    "num_aria_hidden",
    "num_aria_live",
    # "interactive_elements_without_label",

    # New image-accessibility features (2)
    "num_images_without_alt",
    "alt_text_coverage",

    # New heading-accessibility features (2)
    "heading_level_skips",
    "empty_heading_count",

    # New semantic-landmark features (3)
    "has_main_landmark",
    "has_nav_landmark",
    "semantic_element_ratio",

    # New accessible-name and form features (3)
    # "buttons_without_accessible_name",
    # "links_without_accessible_name",
    # "inputs_without_label",
    "unlabelled_button_ratio",
    "unlabelled_link_ratio",
    "unlabelled_input_ratio",


    # New readability and visual-spacing features (2)
    "average_paragraph_word_count",
    "whitespace_ratio",
]

# Assign each above feature the appropriate accessibility direction
FEATURE_DIRECTIONS = {
    # Screenshot-based visual complexity
    "edge_density": "risk",
    "contrast": "risk",
    "colour_variance": "risk",
    # "layout_density": "risk",
    "text_density_proxy": "risk",
    "brightness_variance": "risk",

    # DOM and content complexity
    "dom_depth": "risk",
    "num_links": "risk",
    "num_images": "risk",
    "num_buttons": "risk",
    "num_forms": "risk",
    "num_paragraphs": "risk",
    "num_headings": "risk",
    "word_count": "risk",

    # Advertisement and embedded content
    "num_advertisement_iframes": "risk",
    "num_non_ad_iframes": "risk",

    # ARIA support
    "num_aria_attributes": "protective",
    "aria_attribute_density": "protective",
    "num_aria_label": "protective",
    "num_aria_labelledby": "protective",
    "num_aria_hidden": "protective",
    "num_aria_live": "protective",

    # Existing accessibility issue
    # "interactive_elements_without_label": "risk",

    # Image accessibility
    "num_images_without_alt": "risk",
    "alt_text_coverage": "protective",

    # Heading accessibility
    "heading_level_skips": "risk",
    "empty_heading_count": "risk",

    # Semantic landmarks
    "has_main_landmark": "protective",
    "has_nav_landmark": "protective",
    "semantic_element_ratio": "protective",

    # Accessible names and forms
    # "buttons_without_accessible_name": "risk",
    # "links_without_accessible_name": "risk",
    # "inputs_without_label": "risk",
    "unlabelled_button_ratio": "risk",
    "unlabelled_link_ratio": "risk",
    "unlabelled_input_ratio": "risk",

    # Readability and spacing
    "average_paragraph_word_count": "risk",
    "whitespace_ratio": "protective",
}

# Assign each feature their appropriate weighting
FEATURE_WEIGHTS = {
    # Screenshot-based visual complexity
    "edge_density": 0.045,
    "contrast": 0.030,
    "colour_variance": 0.035,
    # "layout_density": 0.055,
    "text_density_proxy": 0.050,
    "brightness_variance": 0.035,

    # DOM and content complexity
    "dom_depth": 0.035,
    "num_links": 0.035,
    "num_images": 0.020,
    "num_buttons": 0.025,
    "num_forms": 0.015,
    "num_paragraphs": 0.020,
    "num_headings": 0.020,
    "word_count": 0.030,

    # Advertisement and embedded content
    "num_advertisement_iframes": 0.065,
    "num_non_ad_iframes": 0.025,

    # Existing ARIA features
    "num_aria_attributes": 0.015,
    "aria_attribute_density": 0.025,
    "num_aria_label": 0.020,
    "num_aria_labelledby": 0.015,
    "num_aria_hidden": 0.010,
    "num_aria_live": 0.010,
    # "interactive_elements_without_label": 0.025,

    # Image accessibility
    "num_images_without_alt": 0.045,
    "alt_text_coverage": 0.035,

    # Heading accessibility
    "heading_level_skips": 0.035,
    "empty_heading_count": 0.025,

    # Semantic landmarks
    "has_main_landmark": 0.025,
    "has_nav_landmark": 0.020,
    "semantic_element_ratio": 0.025,

    # Accessible names and forms
    # "buttons_without_accessible_name": 0.035,
    # "links_without_accessible_name": 0.035,
    # "inputs_without_label": 0.030,
    "unlabelled_button_ratio": 0.035,
    "unlabelled_link_ratio": 0.035,
    "unlabelled_input_ratio": 0.030,

    # Readability and visual spacing
    "average_paragraph_word_count": 0.025,
    "whitespace_ratio": 0.025,
}

# Validates the feature configuration to ensure nothing is missing or off
def validate_feature_configuration():
    selected = set(FEATURES_FOR_SCORING)
    direction_features = set(FEATURE_DIRECTIONS)
    weighted_features = set(FEATURE_WEIGHTS)

    missing_directions = selected - direction_features
    extra_directions = direction_features - selected
    missing_weights = selected - weighted_features
    extra_weights = weighted_features - selected

    if missing_directions:
        raise ValueError(
            f"Features missing from FEATURE_DIRECTIONS: {sorted(missing_directions)}"
        )

    if extra_directions:
        raise ValueError(
            f"Unexpected features in FEATURE_DIRECTIONS: {sorted(extra_directions)}"
        )

    if missing_weights:
        raise ValueError(
            f"Features missing from FEATURE_WEIGHTS: {sorted(missing_weights)}"
        )

    if extra_weights:
        raise ValueError(
            f"Unexpected features in FEATURE_WEIGHTS: {sorted(extra_weights)}"
        )

    invalid_directions = {
        feature: direction
        for feature, direction in FEATURE_DIRECTIONS.items()
        if direction not in {"risk", "protective"}
    }

    if invalid_directions:
        raise ValueError(f"Invalid feature directions: {invalid_directions}")

    non_positive_weights = {
        feature: weight
        for feature, weight in FEATURE_WEIGHTS.items()
        if weight <= 0
    }

    if non_positive_weights:
        raise ValueError(
            f"All feature weights must be positive: {non_positive_weights}"
        )


validate_feature_configuration()

total_weight = sum(FEATURE_WEIGHTS.values())

FEATURE_WEIGHTS = {
    feature: weight / total_weight
    for feature, weight in FEATURE_WEIGHTS.items()
}

# Initialises the scaler for normalisation
def init_scaler(method):
    if method == "minmax":
        return MinMaxScaler()

    if method == "standard":
        return StandardScaler()

    if method == "robust":
        return RobustScaler()

    raise ValueError(f"Unsupported normalisation method: {method}")

# Function which normalises each feature
def normalise_features(df, features, method="minmax"):
    values = df[features].copy()

    for feature in features:
        values[feature] = pd.to_numeric(values[feature], errors="coerce")
        median_value = values[feature].median()

        if pd.isna(median_value):
            median_value = 0.0

        values[feature] = values[feature].fillna(median_value)

    scaler = init_scaler(method)

    normalised_values = scaler.fit_transform(df[features])

    for i, feature in enumerate(features):
        df[f"{feature}_norm"] = normalised_values[:, i]

    return df


# Generate the accessibility risk score for each entry in the dataset
def compute_risk_score(df, features, weights, directions):
    score = np.zeros(len(df))

    for feature in features:
        norm_feature = f"{feature}_norm"
        weight = weights.get(feature, 0)
        direction = directions.get(feature, "risk")

        if direction == "risk":
            score += df[norm_feature] * weight

        elif direction == "protective":
            score += (1 - df[norm_feature]) * weight

        else:
            raise ValueError(f"Unsupported direction for {feature}: {direction}")

    df["accessibility_risk_score"] = score

    return df

# Assigns the accessibility label bases on score
def assign_accessibility_label(score):
    if score < 0.33:
        return 0  # High accessibility / Low risk
    elif score < 0.66:
        return 1  # Medium accessibility / Medium risk
    else:
        return 2  # Low accessibility / High risk

# Assigns the accessibility text based on score
def assign_accessibility_label_text(score):
    if score < 0.33:
        return "High"
    elif score < 0.66:
        return "Medium"
    else:
        return "Low"


# Display the feature summary following preprocessing
def print_feature_summary():
    print(f"Number of scoring features: {len(FEATURES_FOR_SCORING)}")
    print(f"Normalised weight total: {sum(FEATURE_WEIGHTS.values()):.6f}")

    print("\nFeature configuration:")
    for feature in FEATURES_FOR_SCORING:
        print(
            f"  {feature:<42} "
            f"{FEATURE_DIRECTIONS[feature]:<10} "
            f"{FEATURE_WEIGHTS[feature]:.4f}"
        )


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    print("Loaded dataset:", df.shape)

    missing_features = [
        feature for feature in FEATURES_FOR_SCORING
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    print_feature_summary()

    df = normalise_features(
        df,
        FEATURES_FOR_SCORING,
        method=NORMALISATION_METHOD
    )

    df = compute_risk_score(
        df,
        FEATURES_FOR_SCORING,
        FEATURE_WEIGHTS,
        FEATURE_DIRECTIONS
    )

    df["accessibility_label"] = df["accessibility_risk_score"].apply(
        assign_accessibility_label
    )

    df["accessibility_label_text"] = df["accessibility_risk_score"].apply(
        assign_accessibility_label_text
    )

    df.to_csv(OUTPUT_PATH, index=False)

    # Report distribution and scoring summary
    print("Annotated dataset saved to:", OUTPUT_PATH)

    print("\nLabel distribution:")
    print(df["accessibility_label_text"].value_counts())

    print("\nRisk score summary:")
    print(df["accessibility_risk_score"].describe())

    print("\nLabel percentages:")
    print(df["accessibility_label_text"].value_counts(normalize=True, dropna=False).mul(100).round(2))

    print("\nRisk score summary:")
    print(df["accessibility_risk_score"].describe())


if __name__ == "__main__":
    main()