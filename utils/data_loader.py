import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Defines the necessary features required for prediction generation
FEATURES_FOR_SCORING = [
    # Screenshot-based visual features
    "edge_density",
    "contrast",
    "colour_variance",
    # "layout_density",
    "text_density_proxy",
    "brightness_variance",
    "whitespace_ratio",

    # DOM and content features
    "dom_depth",
    "num_links",
    "num_images",
    "num_buttons",
    "num_forms",
    "num_paragraphs",
    "num_headings",
    "word_count",

    # Advertisement and embedded-content features
    "num_advertisement_iframes",
    "num_non_ad_iframes",

    # ARIA and accessible-naming features
    "num_aria_attributes",
    "aria_attribute_density",
    "num_aria_label",
    "num_aria_labelledby",
    "num_aria_hidden",
    "num_aria_live",
    # "interactive_elements_without_label",

    # Image accessibility features
    "num_images_without_alt",
    "alt_text_coverage",

    # Heading accessibility features
    "heading_level_skips",
    "empty_heading_count",

    # Semantic landmark features
    "has_main_landmark",
    "has_nav_landmark",
    "semantic_element_ratio",

    # Accessible-name and form features
    # "buttons_without_accessible_name",
    # "links_without_accessible_name",
    # "inputs_without_label",
    "unlabelled_button_ratio",
    "unlabelled_link_ratio",
    "unlabelled_input_ratio",

    # Readability feature
    "average_paragraph_word_count",
]

NON_FEATURE_COLUMNS = [
    "id",
    "website",
    "page_type",
    "url",
    "screenshot_path",
    "html_path",
    "num_iframes",
    "accessibility_label",
    "accessibility_label_text",
    "advertisement_iframes",
    "accessibility_risk_score"
]

TARGET_COLUMN = "accessibility_label"

def load_dataset(dataset_path="data/final_dataset.csv"):
    df = pd.read_csv(dataset_path)

    missing_features = [
        feature
        for feature in FEATURES_FOR_SCORING
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "The dataset is missing required assessed features: "
            f"{missing_features}"
        )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"The target column '{TARGET_COLUMN}' is missing from the dataset."
        )

    X = df[FEATURES_FOR_SCORING]
    y = df[TARGET_COLUMN]

    # Ensure every model input is numeric.
    for feature in FEATURES_FOR_SCORING:
        X[feature] = pd.to_numeric(X[feature], errors="coerce")

    print(f"Loaded dataset: {df.shape}")
    print(f"Number of model features: {X.shape[1]}")

    return X, y, FEATURES_FOR_SCORING


def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler