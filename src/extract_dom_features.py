import os
import json
import re
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

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

    ids = {
        tag.get("id")
        for tag in all_elements
        if tag.has_attr("id")
    }

    def count_attr(attr_name):
        return sum(1 for tag in all_elements if tag.has_attr(attr_name))

    def count_broken_aria_refs(attr_name):
        broken_refs = 0

        for tag in all_elements:
            if tag.has_attr(attr_name):
                refs = tag.get(attr_name, "").split()
                broken_refs += sum(1 for ref in refs if ref not in ids)

        return broken_refs

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

    # interactive_without_label = sum(
    #     1
    #     for tag in interactive_elements
    #     if not has_accessible_name(tag, soup)
    # )

    return {
        "has_aria_attributes": int(len(aria_attributes) > 0),
        "num_aria_attributes": len(aria_attributes),
        "aria_attribute_density": safe_ratio(
            len(aria_attributes),
            len(all_elements),
        ),
        "num_aria_label": count_attr("aria-label"),
        "num_aria_labelledby": count_attr("aria-labelledby"),
        "num_aria_hidden": count_attr("aria-hidden"),
        "num_aria_live": count_attr("aria-live"),
        "num_broken_aria_labelledby_refs": count_broken_aria_refs(
            "aria-labelledby"
        ),
        # "interactive_elements_without_label": interactive_without_label,
        # Normalized ratios for ML modeling
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

# NOTE: THIS IS NOT USED, AS THE FEATURES EXTRACTED HERE ARE NOT BEING USED IN THIS VERSION - POSSIBLE INCLUSION FOR FUTURE WORK
# Extracts motion and animation indicators from inline styles, style tags,
# media elements, and reduced-motion settings.
def extract_motion_features(soup):
    # Find inline styles and <style> block contents
    style_tags = soup.find_all("style")
    style_content = " ".join([tag.get_text() for tag in style_tags]).lower()

    inline_styles = [
        tag.get("style", "").lower()
        for tag in soup.find_all(True)
        if tag.has_attr("style")
    ]
    all_css_text = style_content + " " + " ".join(inline_styles)

    # CSS Animations and Transitions
    animation_matches = re.findall(
        r"(animation(-name|-duration|-keyframes)?\s*:|@keyframes)",
        all_css_text,
    )
    transition_matches = re.findall(r"transition\s*:", all_css_text)

    # Dynamic and Autoplay Media
    video_elements = soup.find_all("video")
    autoplay_videos = sum(1 for v in video_elements if v.has_attr("autoplay"))

    # Animated GIFs (approximation via img src)
    gif_images = soup.find_all(
        "img", src=re.compile(r"\.gif($|\?)", flags=re.IGNORECASE)
    )

    # Moving web graphics / dynamic dynamic tags
    marquee_tags = soup.find_all("marquee")

    # Accessibility override check (prefers-reduced-motion)
    has_prefers_reduced_motion = int(
        "prefers-reduced-motion" in all_css_text
    )

    total_motion_triggers = (
        len(animation_matches)
        + len(transition_matches)
        + len(video_elements)
        + len(gif_images)
        + len(marquee_tags)
    )

    return {
        "num_css_animations": len(animation_matches),
        "num_css_transitions": len(transition_matches),
        "num_autoplay_videos": autoplay_videos,
        "num_gif_images": len(gif_images),
        "has_prefers_reduced_motion": has_prefers_reduced_motion,
        "total_motion_triggers": total_motion_triggers,
    }
# Extracts colour variance and palette complexity features from inline styles and CSS properties.
def extract_color_complexity_features(soup):
    style_tags = soup.find_all("style")
    style_content = " ".join([tag.get_text() for tag in style_tags])

    inline_styles = [
        tag.get("style", "")
        for tag in soup.find_all(True)
        if tag.has_attr("style")
    ]
    all_css_text = style_content + " " + " ".join(inline_styles)

    # Regular expressions for Hex, RGB/RGBA, and HSL colors
    hex_colors = re.findall(r"#(?:[0-9a-fA-F]{3}){1,2}\b", all_css_text)
    rgb_colors = re.findall(
        r"rgba?\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*(?:,\s*[\d.]+\s*)?\)",
        all_css_text,
        flags=re.IGNORECASE,
    )
    hsl_colors = re.findall(
        r"hsla?\(\s*\d+\s*,\s*[\d.]%+?\s*,\s*[\d.]%+?\s*(?:,\s*[\d.]+\s*)?\)",
        all_css_text,
        flags=re.IGNORECASE,
    )

    all_raw_colors = hex_colors + rgb_colors + hsl_colors
    unique_colors = set(c.lower() for c in all_raw_colors)

    # Check for visual complexity via gradients
    gradients = re.findall(
        r"(linear-gradient|radial-gradient|conic-gradient)",
        all_css_text,
        flags=re.IGNORECASE,
    )

    return {
        "num_unique_css_colors": len(unique_colors),
        "total_color_declarations": len(all_raw_colors),
        "num_css_gradients": len(gradients),
        "has_high_palette_complexity": int(len(unique_colors) > 15),
    }

# Main DOM function - used to extract all DOM features, using the functions defined above
# and stores them in a dictionary which is later combined with the visual features into a
# larger dictionary
def extract_dom_features(html_path):
    with open(html_path, "r", encoding="utf-8", errors="ignore") as file:
        soup = BeautifulSoup(file, "html.parser")

    text = soup.get_text(separator=" ", strip=True)

    advertisement_elements = []

    for tag in soup.find_all(["iframe", "div"]):
        if is_ad_element(tag):
            advertisement_elements.append(tag)

    advertisement_data = []

    for advertisement in advertisement_elements:
        advertisement_data.append({
            "tag": advertisement.name,
            "src": advertisement.get("src"),
            "aria_label": advertisement.get("aria-label"),
            "class": advertisement.get("class"),
            "id": advertisement.get("id"),
            "text": advertisement.get_text(strip=True)[:100],
        })

    num_iframes = len(soup.find_all("iframe"))
    num_advertisement_iframes = len(advertisement_elements)
    num_non_ad_iframes = num_iframes - num_advertisement_iframes

    features = {
        "num_links": len(soup.find_all("a")),
        "num_images": len(soup.find_all("img")),
        "num_scripts": len(soup.find_all("script")),
        "num_iframes": num_iframes,
        "num_buttons": len(soup.find_all("button")),
        "num_forms": len(soup.find_all("form")),
        "num_paragraphs": len(soup.find_all("p")),
        "num_headings": len(soup.find_all(HEADING_TAGS)),
        "text_length": len(text),
        "word_count": len(text.split()),
        "dom_depth": get_dom_depth(soup),

        # Advertisement iframe features
        "num_advertisement_iframes": num_advertisement_iframes,
        "advertisement_iframes": advertisement_data,
        "num_non_ad_iframes": num_non_ad_iframes,
    }

    features.update(extract_aria_features(soup))

    features.update(extract_image_accessibility_features(soup))
    features.update(extract_heading_accessibility_features(soup))
    features.update(extract_landmark_features(soup))
    # features.update(extract_accessible_name_features(soup))       --- NOT UTILISED
    # features.update(extract_form_accessibility_features(soup))    --- NOT UTILISED
    features.update(extract_readability_features(soup))

    # features.update(extract_motion_features(soup))
    # features.update(extract_color_complexity_features(soup))      --- NOT UTILISED

    return features


def main():
    manifest_path = "manifest.csv"
    output_dir = os.path.join("data", "features", "dom")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    manifest = pd.read_csv(manifest_path)
    all_features = []

    for _, row in tqdm(
        manifest.iterrows(),
        total=len(manifest),
        desc="Processing DOM contents:",
    ):
        page_id = str(row["id"]).zfill(3)
        html_path = row["html_path"]

        if not os.path.exists(html_path):
            print(f"Missing HTML file: {html_path}")
            continue

        features = extract_dom_features(html_path)
        features["id"] = page_id
        features["website"] = row["website"]
        features["page_type"] = row["page_type"]

        full_dir = os.path.join(
            output_dir,
            row["website"],
            row["page_type"],
        )

        if not os.path.exists(full_dir):
            os.makedirs(full_dir)

        json_path = os.path.join(full_dir, f"{page_id}.json")

        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(features, file, indent=4)

        all_features.append(features)

    dataframe = pd.DataFrame(all_features)
    dataframe.to_csv("data/features/dom_features_new_v5.csv", index=False)

    print("DOM feature extraction complete.")


if __name__ == "__main__":
    main()
