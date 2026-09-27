from __future__ import annotations

import cv2
import numpy as np

# 3x3 region grid labels, indexed [row][col]
_ROW_LABELS = ["top", "middle", "bottom"]
_COL_LABELS = ["left", "centre", "right"]

# Threshold above which a cell is considered part of the "hot" region.
HOTSPOT_THRESHOLD = 0.5

# Computes deterministic spatial statistics from a Grad-CAM array
def summarize_heatmap(cam: np.ndarray, threshold: float = HOTSPOT_THRESHOLD) -> dict:
    if cam.ndim != 2:
        raise ValueError(f"Expected a 2D heatmap array, got shape {cam.shape}")

    h, w = cam.shape
    total_weight = cam.sum()

    if total_weight <= 1e-8:
        # Degenerate case: model produced an essentially flat/empty heatmap.
        return {
            "centroid_row_pct": 50.0,
            "centroid_col_pct": 50.0,
            "region_label": "the whole page fairly evenly",
            "coverage_pct": 0.0,
            "concentration": "diffuse",
            "bbox": (0, h - 1, 0, w - 1),
        }

    # Intensity-weighted centroid (i.e. "centre of mass" of the activation).
    row_indices, col_indices = np.indices((h, w))
    centroid_row = (row_indices * cam).sum() / total_weight
    centroid_col = (col_indices * cam).sum() / total_weight

    centroid_row_pct = float(centroid_row / (h - 1) * 100)
    centroid_col_pct = float(centroid_col / (w - 1) * 100)

    row_bucket = min(int(centroid_row_pct // (100 / 3)), 2)
    col_bucket = min(int(centroid_col_pct // (100 / 3)), 2)
    region_label = f"{_ROW_LABELS[row_bucket]}-{_COL_LABELS[col_bucket]}"

    hot_mask = cam >= threshold
    coverage_pct = float(hot_mask.mean() * 100)

    if hot_mask.any():
        rows_hot, cols_hot = np.where(hot_mask)
        bbox = (int(rows_hot.min()), int(rows_hot.max()), int(cols_hot.min()), int(cols_hot.max()))
    else:
        bbox = (int(centroid_row), int(centroid_row), int(centroid_col), int(centroid_col))

    # A "focused" hotspot covers a small fraction of the page; a "broad" one covers a lot.
    concentration = "focused" if coverage_pct <= 15 else ("broad" if coverage_pct <= 40 else "very broad")

    return {
        "centroid_row_pct": round(centroid_row_pct, 1),
        "centroid_col_pct": round(centroid_col_pct, 1),
        "region_label": region_label,
        "coverage_pct": round(coverage_pct, 1),
        "concentration": concentration,
        "bbox": bbox,
    }

# Splits the Grad-CAM hotspot into separate connected regions
def find_hotspot_regions(cam: np.ndarray, threshold: float = HOTSPOT_THRESHOLD, max_regions: int = 8, min_area_px: int = 6,) -> list[dict]:
    h, w = cam.shape
    mask = (cam >= threshold).astype(np.uint8)
    if mask.sum() == 0:
        return []

    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    regions = []
    for label_id in range(1, num_labels):  # label 0 is background
        area_px = int(stats[label_id, cv2.CC_STAT_AREA])
        if area_px < min_area_px:
            continue

        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        bw = int(stats[label_id, cv2.CC_STAT_WIDTH])
        bh = int(stats[label_id, cv2.CC_STAT_HEIGHT])

        blob_mask = labels_im == label_id
        mass = float(cam[blob_mask].sum())

        cx, cy = centroids[label_id]
        centroid_row_pct = float(cy / (h - 1) * 100)
        centroid_col_pct = float(cx / (w - 1) * 100)
        row_bucket = min(int(centroid_row_pct // (100 / 3)), 2)
        col_bucket = min(int(centroid_col_pct // (100 / 3)), 2)
        region_label = f"{_ROW_LABELS[row_bucket]}-{_COL_LABELS[col_bucket]}"

        regions.append({
            "bbox": (y, y + bh - 1, x, x + bw - 1),
            "centroid_row_pct": round(centroid_row_pct, 1),
            "centroid_col_pct": round(centroid_col_pct, 1),
            "region_label": region_label,
            "coverage_pct": round(area_px / (h * w) * 100, 2),
            "mass": mass,
        })

    total_mass = sum(r["mass"] for r in regions) or 1.0
    for r in regions:
        r["mass_share_pct"] = round(r["mass"] / total_mass * 100, 1)

    regions.sort(key=lambda r: r["mass"], reverse=True)
    return regions[:max_regions]

# Takes a list of hotspot regions and generates a natural language explanation.
def regions_to_sentence(regions: list[dict], predicted_label: str) -> str:
    if not regions:
        return "The model's visual attention was too diffuse to identify a specific hotspot."

    if len(regions) == 1:
        r = regions[0]
        return (
            f"The model's visual attention (Grad-CAM) was concentrated in the "
            f"**{r['region_label']}** area of the page (around {r['coverage_pct']}% of the "
            f"screenshot area), which was the strongest visual driver behind the "
            f"**{predicted_label}** prediction."
        )

    parts = [f"the **{r['region_label']}** area ({r['mass_share_pct']}% of total attention)" for r in regions]
    joined = ", ".join(parts[:-1]) + f", and {parts[-1]}"
    return (
        f"The model's visual attention (Grad-CAM) was split across {len(regions)} separate "
        f"areas of the page, in order of influence: {joined}. Together, these were the "
        f"strongest visual drivers behind the **{predicted_label}** prediction."
    )

# Rescales a heatmap-grid bounding box into the screenshot's own pixel coordinates
def _bbox_to_page_pixels(bbox, heatmap_shape, screenshot_size):
    h, w = heatmap_shape
    shot_w, shot_h = screenshot_size
    row_min, row_max, col_min, col_max = bbox
    x_scale = shot_w / w
    y_scale = shot_h / h
    return (
        col_min * x_scale,
        row_min * y_scale,
        col_max * x_scale,
        row_max * y_scale,
    )  # (x_min, y_min, x_max, y_max) in screenshot pixel coordinates


def gen_overlap_area(box_a, box_b) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)

# Ranks real DOM elements by how much they overlap a Grad-CAM hotspot region
def match_elements_to_hotspot(bbox: tuple[int, int, int, int], heatmap_shape: tuple[int, int], elements: list[dict], screenshot_size: tuple[float, float], top_k: int = 3, max_area_ratio: float = 0.35, min_overlap_ratio: float = 0.3, dedupe_iou: float = 0.85,) -> list[dict]:
    hotspot_px = _bbox_to_page_pixels(bbox, heatmap_shape, screenshot_size)
    hotspot_area = max(1.0, (hotspot_px[2] - hotspot_px[0]) * (hotspot_px[3] - hotspot_px[1]))
    screenshot_area = max(1.0, screenshot_size[0] * screenshot_size[1])

    scored = []
    for el in elements:
        el_area = max(1.0, el["width"] * el["height"])
        if el_area / screenshot_area > max_area_ratio:
            continue  # too big to be a meaningful "location" - skip wrapper-like elements

        el_box = (el["x"], el["y"], el["x"] + el["width"], el["y"] + el["height"])
        overlap = gen_overlap_area(hotspot_px, el_box)
        if overlap <= 0:
            continue

        overlap_ratio = overlap / min(hotspot_area, el_area)
        if overlap_ratio < min_overlap_ratio:
            continue

        scored.append({**el, "overlap_area": overlap, "overlap_ratio": overlap_ratio, "element_area": el_area})

    # Prefer the strongest, most specific (smallest) matches first.
    scored.sort(key=lambda e: (-e["overlap_ratio"], e["element_area"]))

    selected: list[dict] = []
    for candidate in scored:
        c_box = (
            candidate["x"], candidate["y"],
            candidate["x"] + candidate["width"], candidate["y"] + candidate["height"],
        )
        is_duplicate = False
        for chosen in selected:
            chosen_box = (
                chosen["x"], chosen["y"],
                chosen["x"] + chosen["width"], chosen["y"] + chosen["height"],
            )
            inter = gen_overlap_area(c_box, chosen_box)
            union = candidate["element_area"] + chosen["element_area"] - inter
            iou = inter / union if union > 0 else 0
            if iou >= dedupe_iou:
                is_duplicate = True
                break
        if not is_duplicate:
            selected.append(candidate)
        if len(selected) >= top_k:
            break

    return selected


def describe_element(el: dict) -> str:
    label = el.get("aria_label") or el.get("text") or el.get("id") or el.get("class_name")
    if label:
        return f"a `<{el['tag']}>` element (\"{label}\")"
    return f"a `<{el['tag']}>` element"

# Takes ranked DOM element matches and generates a human-readable explanation from these matches
def elements_to_sentence(matches: list[dict]) -> str | None:
    if not matches:
        return None
    described = [describe_element(el) for el in matches]
    if len(described) == 1:
        joined = described[0]
    else:
        joined = ", ".join(described[:-1]) + f", and {described[-1]}"
    return f"Within that region, the model's attention most closely overlapped with {joined}."

# Draws Grad-CAM hotspot region(s) and any matched DOM elements onto the screenshot
def draw_gradcam_debug_overlay(
    screenshot,
    regions: list[dict],
    heatmap_shape: tuple[int, int],
    matches_by_region: list[list[dict]] | None = None,
):
    from PIL import ImageDraw

    annotated = screenshot.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)

    region_palette = [(255, 0, 0), (255, 140, 0), (200, 0, 200), (0, 150, 200)]
    element_palette = [(0, 120, 255), (0, 200, 100), (255, 165, 0)]

    for region_idx, region in enumerate(regions):
        region_color = region_palette[region_idx % len(region_palette)]
        hotspot_px = _bbox_to_page_pixels(region["bbox"], heatmap_shape, annotated.size)
        draw.rectangle(hotspot_px, outline=region_color, width=4)
        rank_label = f"Hotspot #{region_idx + 1}"
        if "mass_share_pct" in region:
            rank_label += f" ({region['mass_share_pct']}%)"
        draw.text((hotspot_px[0] + 4, max(0, hotspot_px[1] - 18)), rank_label, fill=region_color)

        # region_matches = (matches_by_region or [])[region_idx] if matches_by_region else []
        # for el_idx, el in enumerate(region_matches):
        #     box = (el["x"], el["y"], el["x"] + el["width"], el["y"] + el["height"])
        #     el_color = element_palette[el_idx % len(element_palette)]
        #     draw.rectangle(box, outline=el_color, width=2)
        #     draw.text((box[0] + 4, max(0, box[1] - 14)), f"<{el['tag']}>", fill=el_color)

    return annotated

# Turns the summary dictionary into a human-readable summary
def heatmap_summary_to_sentence(summary: dict, predicted_label: str) -> str:
    return (
        f"The model's visual attention (Grad-CAM) was most {summary['concentration']}ly "
        f"concentrated in the **{summary['region_label']}** area of the page "
        f"(around {summary['coverage_pct']}% of the screenshot area), "
        f"which was the strongest visual driver behind the **{predicted_label}** prediction."
    )