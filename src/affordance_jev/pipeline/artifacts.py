"""Model-independent candidate preparation and RGB-aligned result artifacts."""
from pathlib import Path
import time

import numpy as np
from PIL import Image

from ..common import read, write, sha, bbox, load_mask, save_mask, validate_classes
from ..decision.regions import request_for, decode, merge
from ..segmentation.partition import partition, remove_small
from ..visualization import overlay, sheet, report


def prepare_regions(job):
    started = time.perf_counter()
    out = Path(job["output"])
    config = read(out / "config_snapshot.json")
    definitions = read(out / "affordances_snapshot.json")
    rgb = Image.open(out / "rgb.png").convert("RGB")
    fg = load_mask(out / "foreground/object_mask.png", (rgb.height, rgb.width))
    crop = read(out / "foreground/metadata.json")["crop_xyxy"]
    with np.load(out / "raw/proposals.npz", allow_pickle=False) as raw:
        masks, quality = raw["masks"], raw["qualities"]
    regions, sources, trace = partition(masks, quality, fg, config["partition"])
    folder = out / "regions"
    folder.mkdir(parents=True, exist_ok=True)
    write(folder / "filter_trace.json", trace)
    filtered_panels = []
    for row in trace["filter"]:
        raw_id = row["raw_id"]
        clipped = masks[raw_id] & fg
        save_mask(out / f"raw/parts_clipped/{raw_id:03}.png", clipped)
        if row["rejection"] is None:
            cleaned = remove_small(clipped, trace["minimum_component_pixels"])
            save_mask(folder / f"filtered/{raw_id:03}.png", cleaned)
            filtered_panels.append((f"raw {raw_id} / q={quality[raw_id]:.3f}",
                                    overlay(rgb, [cleaned], labels=[str(raw_id)]).crop(crop)))
    sheet(filtered_panels, folder / "filtered_candidates.jpg", columns=4)
    label_map = np.zeros(fg.shape, np.uint16)
    records, panels = [], []
    for i, (mask, raw_id) in enumerate(zip(regions, sources), 1):
        relative = f"regions/masks/{i:02}.png"
        save_mask(out / relative, mask)
        label_map[mask] = i
        records.append({"id": i, "raw_id": raw_id, "mask": relative, "mask_sha256": sha(out / relative),
                        "quality": float(quality[raw_id]), "area": int(mask.sum()), "bbox_xyxy": bbox(mask)})
        im = overlay(rgb, [mask], labels=[str(i)])
        im.save(folder / f"region_{i:02}.png")
        panels.append((f"Region {i}", im.crop(crop)))
    Image.fromarray(label_map).save(folder / "label_map.png")
    save_mask(folder / "unassigned_foreground.png", fg & (label_map == 0))
    numbered = overlay(rgb, regions)
    numbered.save(folder / "numbered_full.png")
    numbered.crop(crop).save(folder / "numbered_crop.png")
    sheet(panels, folder / "region_sheet.jpg", columns=3)
    write(out / "candidates.json", {
        "coordinate_system": "original RGB pixels", "size_wh": list(rgb.size),
        "candidates": records, "foreground_pixels": int(fg.sum()),
        "unassigned_foreground_pixels": int((fg & (label_map == 0)).sum()),
        "postprocessing": "specificity-priority pixel assignment; not official PartGLEE panoptic output",
    })
    request = request_for(records, definitions, config["valen"]["include_region_sheet"])
    write(out / "valen_request.json", request)
    assets = [
        "rgb.png", "foreground/object_mask.png", "foreground/rgb_crop.png",
        "regions/numbered_crop.png", "regions/region_sheet.jpg", "config_snapshot.json",
        "affordances_snapshot.json", "candidates.json", "valen_request.json",
    ] + [r["mask"] for r in records]
    write(out / "prepared_assets.json", {name: sha(out / name) for name in assets})
    write(out / "run.json", {"status": "prepared", "reference_annotations_used": False})
    report(out, "候选已就绪")
    write(out / "prepare_runtime.json", {"candidate_postprocess_and_save_seconds": time.perf_counter() - started})


def verify_prepared(out):
    out = Path(out)
    for name, digest in read(out / "prepared_assets.json").items():
        if sha(out / name) != digest:
            raise ValueError(f"Prepared asset changed: {name}")


def merge_results(out):
    out = Path(out)
    started = time.perf_counter()
    verify_prepared(out)
    response = read(out / "valen_response.json")
    if response.get("request_sha256") != sha(out / "valen_request.json"):
        raise ValueError("Response belongs to another request")
    candidates = read(out / "candidates.json")["candidates"]
    definitions = read(out / "affordances_snapshot.json")
    config = read(out / "config_snapshot.json")
    classes = validate_classes(definitions)
    rgb = Image.open(out / "rgb.png").convert("RGB")
    fg = load_mask(out / "foreground/object_mask.png", (rgb.height, rgb.width))
    regions = [load_mask(out / r["mask"], fg.shape) for r in candidates]
    decisions = decode(response, candidates, definitions)
    merged, class_map = merge(regions, decisions, fg, definitions, config["partition"].get("whole_object_fraction"))
    result = out / "result"
    result.mkdir(exist_ok=True)
    records, panels = [], []
    crop = read(out / "foreground/metadata.json")["crop_xyxy"]
    for cls in classes:
        name = cls["name"]
        save_mask(result / f"masks/{name}.png", merged[name])
        im = overlay(rgb, [merged[name]], colors=[cls["color"]], labels=[name])
        im.save(result / f"{name}_overlay.png")
        records.append({
            "class": name, "id": cls["id"], "mask": f"masks/{name}.png",
            "pixels": int(merged[name].sum()),
            "region_ids": [d["region_id"] for d in decisions if d["final_choice"] == name],
        })
        panels.append((name, im.crop(crop)))
    Image.fromarray(class_map).save(result / "class_map.png")
    colors = {cls["name"]: cls["color"] for cls in classes}
    overlay(rgb, regions, colors=[colors[d["final_choice"]] for d in decisions],
            labels=[f"{d['region_id']}:{d['final_choice']}" for d in decisions]).save(result / "classified_regions.png")
    overlay(rgb, [merged[c["name"]] for c in classes],
            colors=[c["color"] for c in classes], labels=[c["name"] for c in classes]).save(result / "affordance_overlay.png")
    sheet(panels, result / "class_sheet.jpg", columns=4)
    write(out / "decisions.json", {"mode": "single_label", "regions": decisions,
          "note": "Probabilities are model scores, not measured accuracy. none remains explicit although its native map ID is 0."})
    write(result / "masks.json", {"size_wh": list(rgb.size), "classes": records, "class_map": "class_map.png",
          "coordinate_system": "original RGB pixels", "background_unassigned_none_id": 0,
          "none_mask": "masks/none.png", "unassigned_foreground_mask": "../regions/unassigned_foreground.png"})
    saved = np.stack([load_mask(result / r["mask"], fg.shape) for r in records])
    coverage = saved.sum(0)
    candidate_union = np.logical_or.reduce(regions) if regions else np.zeros(fg.shape, bool)
    expected_map = np.zeros(fg.shape, np.uint8)
    for cls, mask in zip(classes, saved):
        expected_map[mask] = cls["id"]
    audit = {
        "size_wh": list(rgb.size), "region_count": len(regions),
        "overlap_pixels": int((coverage > 1).sum()),
        "outside_object_pixels": int(((coverage > 0) & ~fg).sum()),
        "class_map_matches_masks": bool(np.array_equal(expected_map, class_map)),
        "all_candidate_pixels_accounted_for": bool(np.array_equal(coverage > 0, candidate_union)),
        "reference_annotations_used": False, "whole_object_fraction": config["partition"].get("whole_object_fraction"),
        "predicted_foreground_coverage": float(candidate_union.sum() / max(1, fg.sum())),
    }
    write(out / "audit.json", audit)
    if audit["overlap_pixels"] or audit["outside_object_pixels"] or not audit["class_map_matches_masks"] or not audit["all_candidate_pixels_accounted_for"]:
        raise AssertionError(audit)
    write(out / "run.json", {"status": "complete", "reference_annotations_used": False})
    report(out, "推理完成；未使用 GT")
    timing = {
        "merge_and_save_seconds": time.perf_counter() - started,
        "partglee": read(out / "partglee_runtime.json"),
        "prepare": read(out / "prepare_runtime.json"),
        "valen": read(out / "valen_runtime.json"),
        "note": "Stage timings from staged batch execution. Not a live RGB-to-mask latency or real-time FPS.",
    }
    write(out / "timings.json", timing)
    report(out, "推理完成；未使用 GT", timing)
