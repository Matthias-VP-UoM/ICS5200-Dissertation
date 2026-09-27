import cv2
import numpy as np
import re
from bs4 import BeautifulSoup

# Define both heading and semantic tags for feature extraction
HEADING_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6"]
SEMANTIC_TAGS = {
    "main",
    "nav",
    "header",
    "footer",
    "section",
    "article",
    "aside",
    "form",
    "figure",
    "figcaption",
    "details",
    "summary",
}

# Define features used for generating prediction
FEATURE_COLUMNS = [
    # Screenshot-based visual features
    "edge_density",
    "contrast",
    "colour_variance",
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


## Helper functions used to compute and extract the necessary DOM features

# Checks whether a given element is an advertisement
def is_ad_element(tag):
    class_values = " ".join(tag.get("class", [])).lower()
    aria_label = tag.get("aria-label", "").lower()

    return (
        aria_label == "advertisement"
        or "tmiads" in class_values
        or class_values.startswith("ad ")
    )

# Computes and returns the DOM depth of the given website
def get_dom_depth(element, current_depth=0):
    if not hasattr(element, "children"):
        return current_depth

    child_depths = [
        get_dom_depth(child, current_depth + 1)
        for child in element.children
        if getattr(child, "name", None) is not None
    ]

    return max(child_depths, default=current_depth)


def normalise_text(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0

# Retrieves the aria-labelledby tag value
def get_aria_labelledby_text(tag, soup):
    referenced_text = []
    for element_id in tag.get("aria-labelledby", "").split():
        referenced_element = soup.find(id=element_id)
        if referenced_element is not None:
            text = normalise_text(referenced_element.get_text(" ", strip=True))
            if text:
                referenced_text.append(text)
    return " ".join(referenced_text)

# Checks whether an element tag has an accessible name
def has_accessible_name(tag, soup):
    has_visible_text = bool(normalise_text(tag.get_text(" ", strip=True)))
    has_aria_label = bool(normalise_text(tag.get("aria-label")))
    has_aria_labelledby = bool(get_aria_labelledby_text(tag, soup))
    has_title = bool(normalise_text(tag.get("title")))

    image = tag.find("img")
    has_child_image_alt = bool(
        image is not None and normalise_text(image.get("alt"))
    )

    has_input_value = False
    if tag.name == "input":
        input_type = tag.get("type", "text").lower()
        if input_type in {"button", "submit", "reset"}:
            has_input_value = bool(normalise_text(tag.get("value")))
        elif input_type == "image":
            has_input_value = bool(normalise_text(tag.get("alt")))

    return any([
        has_visible_text,
        has_aria_label,
        has_aria_labelledby,
        has_title,
        has_child_image_alt,
        has_input_value,
    ])

# Checks whether an input has an assigned label (possibly ARIA)
def input_has_label(tag, soup):
    if tag.name == "input" and tag.get("type", "text").lower() == "hidden":
        return True

    if normalise_text(tag.get("aria-label")):
        return True

    if get_aria_labelledby_text(tag, soup):
        return True

    element_id = tag.get("id")
    if element_id:
        explicit_label = soup.find("label", attrs={"for": element_id})
        if explicit_label is not None and normalise_text(
            explicit_label.get_text(" ", strip=True)
        ):
            return True

    parent_label = tag.find_parent("label")
    if parent_label is not None and normalise_text(
        parent_label.get_text(" ", strip=True)
    ):
        return True

    if tag.name == "input":
        input_type = tag.get("type", "text").lower()
        if input_type in {"button", "submit", "reset"}:
            return bool(normalise_text(tag.get("value")))
        if input_type == "image":
            return bool(normalise_text(tag.get("alt")))

    return False

# Checks if an element has an accessible name through visible text,
# aria-label, aria-labelledby, title, or alt attributes.
def check_element_labelled(tag):
    has_visible_text = bool(tag.get_text(strip=True))
    has_aria_label = bool(tag.get("aria-label", "").strip())
    has_aria_labelledby = tag.has_attr("aria-labelledby")
    has_title = bool(tag.get("title", "").strip())
    has_alt = bool(tag.get("alt", "").strip())

    return any([
        has_visible_text,
        has_aria_label,
        has_aria_labelledby,
        has_title,
        has_alt
    ])

# Extracts all the ARIA features which are defined above
def extract_aria_features(soup):
    all_elements = soup.find_all()

    aria_attributes = [
        attr
        for tag in all_elements
        for attr in tag.attrs
        if attr.startswith("aria-")
    ]

    def count_attr(attr_name):
        return sum(1 for tag in all_elements if tag.has_attr(attr_name))
    
    # Granular interactive element parsing
    buttons = soup.find_all("button")
    links = soup.find_all("a")
    inputs = soup.find_all(["input", "select", "textarea"])

    # Total counts
    total_buttons = len(buttons)
    total_links = len(links)
    total_inputs = len(inputs)

    # Unlabelled counts
    unlabelled_buttons = sum(1 for b in buttons if not check_element_labelled(b))
    unlabelled_links = sum(1 for l in links if not check_element_labelled(l))
    unlabelled_inputs = sum(1 for i in inputs if not check_element_labelled(i))

    # Safe ratio calculation (handles division by zero)
    unlabelled_button_ratio = unlabelled_buttons / total_buttons if total_buttons > 0 else 0.0
    unlabelled_link_ratio = unlabelled_links / total_links if total_links > 0 else 0.0
    unlabelled_input_ratio = unlabelled_inputs / total_inputs if total_inputs > 0 else 0.0

    # interactive_elements = soup.find_all(
    #     ["a", "button", "input", "select", "textarea"]
    # )

    # interactive_without_label = sum(
    #     1
    #     for tag in interactive_elements
    #     if not has_accessible_name(tag, soup)
    # )

    return {
        "num_aria_attributes": len(aria_attributes),
        "aria_attribute_density": safe_ratio(
            len(aria_attributes),
            len(all_elements),
        ),
        "num_aria_label": count_attr("aria-label"),
        "num_aria_labelledby": count_attr("aria-labelledby"),
        "num_aria_hidden": count_attr("aria-hidden"),
        "num_aria_live": count_attr("aria-live"),
        # "interactive_elements_without_label": interactive_without_label,
        "unlabelled_button_ratio": round(unlabelled_button_ratio, 4),
        "unlabelled_link_ratio": round(unlabelled_link_ratio, 4),
        "unlabelled_input_ratio": round(unlabelled_input_ratio, 4),
    }

# Extracts DOM accessibility features related to imagery (i.e. num_images_without_alt and alt_text_coverage)
def extract_image_accessibility_features(soup):
    images = soup.find_all("img")
    images_without_alt = sum(1 for image in images if not image.has_attr("alt"))
    images_with_alt = len(images) - images_without_alt

    return {
        "num_images_without_alt": images_without_alt,
        "alt_text_coverage": safe_ratio(images_with_alt, len(images)),
    }

# Extracts heading accessibility features defined above (i.e. heading_level_skips and empty_heading_count)
def extract_heading_accessibility_features(soup):
    headings = soup.find_all(HEADING_TAGS)

    heading_levels = [
        int(heading.name[1])
        for heading in headings
        if heading.name and len(heading.name) == 2
    ]

    heading_level_skips = sum(
        1
        for previous_level, current_level in zip(
            heading_levels,
            heading_levels[1:],
        )
        if current_level > previous_level + 1
    )

    empty_heading_count = sum(
        1
        for heading in headings
        if not normalise_text(heading.get_text(" ", strip=True))
    )

    return {
        "heading_level_skips": heading_level_skips,
        "empty_heading_count": empty_heading_count,
    }

# Extracts all the landmark features defined above (i.e. has_main_landmark, has_nav_landmark and semantic_element_ratio)
def extract_landmark_features(soup):
    all_elements = soup.find_all()

    semantic_elements = [
        tag
        for tag in all_elements
        if tag.name in SEMANTIC_TAGS or tag.has_attr("role")
    ]

    has_main_landmark = int(
        soup.find("main") is not None
        or soup.find(attrs={"role": "main"}) is not None
    )

    has_nav_landmark = int(
        soup.find("nav") is not None
        or soup.find(attrs={"role": "navigation"}) is not None
    )

    return {
        "has_main_landmark": has_main_landmark,
        "has_nav_landmark": has_nav_landmark,
        "semantic_element_ratio": safe_ratio(
            len(semantic_elements),
            len(all_elements),
        ),
    }

# NOTE: THIS IS NOT USED, AS THE FEATURES EXTRACTED HERE ARE NO LONGER USED
# Extracts all buttons and hyperlinks that do not have an accessible name
def extract_accessible_name_features(soup):
    buttons = list(soup.find_all("button"))
    buttons.extend(
        soup.find_all(
            "input",
            attrs={
                "type": re.compile(
                    r"^(button|submit|reset|image)$",
                    flags=re.IGNORECASE,
                )
            },
        )
    )

    links = soup.find_all("a")

    return {
        "buttons_without_accessible_name": sum(
            1 for button in buttons if not has_accessible_name(button, soup)
        ),
        "links_without_accessible_name": sum(
            1 for link in links if not has_accessible_name(link, soup)
        ),
    }

# NOTE: THIS IS NOT USED, AS THE FEATURES EXTRACTED HERE ARE NO LONGER USED
# Extracts all input elements that do not have a label
def extract_form_accessibility_features(soup):
    controls = soup.find_all(["input", "select", "textarea"])

    return {
        "inputs_without_label": sum(
            1 for control in controls if not input_has_label(control, soup)
        )
    }

# Extracts any readibility features which are defined above (i.e. average_paragraph_word_count)
def extract_readability_features(soup):
    paragraph_word_counts = []

    for paragraph in soup.find_all("p"):
        text = normalise_text(paragraph.get_text(" ", strip=True))

        if text:
            words = re.findall(r"\b[\w'’-]+\b", text, flags=re.UNICODE)

            if words:
                paragraph_word_counts.append(len(words))

    average_paragraph_word_count = (
        sum(paragraph_word_counts) / len(paragraph_word_counts)
        if paragraph_word_counts
        else 0.0
    )

    return {
        "average_paragraph_word_count": average_paragraph_word_count,
    }

# Main DOM function - used to extract all DOM features, using the functions defined above
# and stores them in a dictionary which is later combined with the visual features into a
# larger dictionary
def extract_dom_features(html_path):
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")

    text = soup.get_text(separator=" ", strip=True)

    advertisement_elements = [
        tag for tag in soup.find_all(["iframe", "div"]) if is_ad_element(tag)
    ]

    num_iframes = len(soup.find_all("iframe"))
    num_advertisement_iframes = len(advertisement_elements)
    num_non_ad_iframes = max(0, num_iframes - num_advertisement_iframes)

    features = {
        "dom_depth": get_dom_depth(soup),
        "num_links": len(soup.find_all("a")),
        "num_images": len(soup.find_all("img")),
        "num_buttons": len(soup.find_all("button")),
        "num_forms": len(soup.find_all("form")),
        "num_paragraphs": len(soup.find_all("p")),
        "num_headings": len(soup.find_all(HEADING_TAGS)),
        "word_count": len(text.split()),
        "num_advertisement_iframes": num_advertisement_iframes,
        "num_non_ad_iframes": num_non_ad_iframes,
    }

    # Integrate feature extraction modules
    features.update(extract_aria_features(soup))
    features.update(extract_image_accessibility_features(soup))
    features.update(extract_heading_accessibility_features(soup))
    features.update(extract_landmark_features(soup))
    # features.update(extract_accessible_name_features(soup))
    # features.update(extract_form_accessibility_features(soup))
    features.update(extract_readability_features(soup))

    return features


## Helper functions used to compute and extract the necessary visual features

# Calculates edge density within a given screenshot
def compute_edge_density(gray):
    edges = cv2.Canny(gray, 100, 200)
    return np.sum(edges > 0) / edges.size


def compute_contrast(gray):
    return float(np.std(gray))


def compute_colour_variance(image):
    return float(np.var(image))


def compute_text_density_proxy(gray):
    edges = cv2.Canny(gray, 50, 150)
    return np.sum(edges > 0) / edges.size


def compute_whitespace_ratio(gray):
    """Measures the proportion of near-white background pixels."""
    threshold = 245
    return float(np.sum(gray >= threshold) / gray.size)

# Main imagery function - used to extract all visual features, using the functions defined above
# and stores them in a dictionary which is later combined with the DOM features into a
# larger dictionary
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
        "text_density_proxy": compute_text_density_proxy(gray),
        "brightness_variance": float(np.var(gray)),
        "whitespace_ratio": compute_whitespace_ratio(gray),
    }

    return features


# Main script function - extracts all DOm and imagery features using the other 2 main functions
# and stores them into a large dictionary which is returned to the program
def extract_live_features(screenshot_path, html_path):
    dom_features = extract_dom_features(html_path)
    image_features = extract_image_features(screenshot_path)

    all_extracted_features = {
        **image_features,
        **dom_features,
    }

    # Guarantee key order aligns strictly with FEATURE_COLUMNS
    ordered_features = {col: all_extracted_features[col] for col in FEATURE_COLUMNS}

    return ordered_features