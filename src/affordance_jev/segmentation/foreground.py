"""Explicit RGB-camera prior for one central, fully visible target object."""
import math
from ..common import bbox


def select_object(masks, qualities, config):
    if config["selection"] != "central_complete_object":
        raise ValueError("Unknown foreground selection mode")
    h, w = masks.shape[1:]
    if config["center_sigma"] <= 0 or not 0 <= config["border_penalty"] <= 1:
        raise ValueError("Invalid foreground ranking parameters")
    trace = []
    for i, (mask, quality) in enumerate(zip(masks, qualities)):
        pixels, box = int(mask.sum()), bbox(mask)
        row = {"raw_id": i, "quality": quality, "pixels": pixels, "bbox_xyxy": box, "rank": 0.0, "eligible": False}
        if box and pixels >= config["minimum_object_pixels"] and quality >= config["minimum_object_score"]:
            x0, y0, x1, y1 = box
            distance2 = (((x0+x1)/2-w/2)/(w/2))**2 + (((y0+y1)/2-h/2)/(h/2))**2
            center_weight = math.exp(-distance2/(2*config["center_sigma"]**2))
            border = x0 == 0 or y0 == 0 or x1 == w or y1 == h
            rank = quality * math.sqrt(pixels/(h*w)) * center_weight * (config["border_penalty"] if border else 1)
            row.update(rank=rank, eligible=True, touches_image_border=border, center_weight=center_weight)
        trace.append(row)
    eligible = [row for row in trace if row["eligible"]]
    if not eligible:
        raise ValueError("No reliable PartGLEE object proposal; inspect raw/object_candidates.jpg or provide OBJECT_MASK")
    selected = max(eligible, key=lambda row: row["rank"])["raw_id"]
    return selected, trace
