"""Convert overlapping part proposals to a small, disjoint set (no GT input)."""
import cv2
import numpy as np


def remove_small(mask, minimum):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= minimum]
    return np.isin(labels, keep)


def partition(raw_masks, qualities, foreground, config):
    masks = np.asarray(raw_masks, dtype=bool)
    quality = np.asarray(qualities, dtype=np.float32)
    if masks.ndim != 3 or masks.shape[1:] != foreground.shape or len(masks) != len(quality):
        raise ValueError("Invalid proposal/quality/foreground shapes")
    if not np.isfinite(quality).all() or (quality < 0).any():
        raise ValueError("Invalid PartGLEE quality values")
    area = int(foreground.sum())
    if not area:
        raise ValueError("Empty object foreground")
    cap = int(config["max_regions"])
    if not 1 <= cap <= 255:
        raise ValueError("max_regions must be 1..255")
    minimum = max(int(config["minimum_component_pixels"]), int(area * config["minimum_component_fraction"]))
    whole_fraction = config.get("whole_object_fraction")
    if minimum < 1 or (whole_fraction is not None and not 0 < whole_fraction <= 1):
        raise ValueError("Invalid area thresholds")
    cleaned, sources, filtered = [], [], []
    for i, original in enumerate(masks):
        mask = remove_small(original & foreground, minimum)
        pixels = int(mask.sum())
        reason = None
        if quality[i] < config["minimum_quality"]:
            reason = "low_quality"
        elif pixels < minimum:
            reason = "small_after_clipping"
        elif whole_fraction is not None and pixels / area >= whole_fraction:
            reason = "near_whole_object"
        filtered.append({"raw_id": i, "quality": float(quality[i]), "clipped_cleaned_area": pixels, "rejection": reason})
        if reason is None:
            cleaned.append(mask)
            sources.append(i)
    history = []
    if not cleaned:
        return [], [], {"minimum_component_pixels": minimum, "filter": filtered, "assignment": history}
    cleaned = np.stack(cleaned)
    q = quality[sources]
    priority = q / (cleaned.sum((1, 2)) / area) ** float(config["specificity_exponent"])
    selected = list(range(len(cleaned)))
    regions = []
    while selected:
        scores = cleaned[selected] * priority[selected, None, None]
        labels = scores.argmax(0) + 1
        labels[scores.max(0) <= 0] = 0
        regions = [remove_small(labels == i + 1, minimum) for i in range(len(selected))]
        valid = [i for i, mask in enumerate(regions) if mask.sum() >= minimum]
        if len(valid) != len(selected):
            history.append({"reason": "small_after_assignment", "removed_raw_ids": [sources[selected[i]] for i in range(len(selected)) if i not in valid]})
            selected = [selected[i] for i in valid]
            regions = []
            continue
        if len(selected) > cap:
            importance = np.array([m.sum() for m in regions]) * np.sqrt(q[selected])
            index = int(importance.argmin())
            history.append({"reason": "region_cap", "removed_raw_ids": [sources[selected[index]]]})
            selected.pop(index)
            regions = []
            continue
        break
    source_ids = [sources[i] for i in selected]
    if regions and np.stack(regions).sum(0).max() > 1:
        raise AssertionError("Partition masks overlap")
    return regions, source_ids, {"minimum_component_pixels": minimum, "filter": filtered, "assignment": history,
                                "priority_formula": "quality / (area / foreground_area) ** specificity_exponent"}
