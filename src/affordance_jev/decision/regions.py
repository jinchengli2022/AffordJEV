"""Functional decisions over disjoint candidates; no annotation access."""
import math
import numpy as np

from ..common import validate_classes


def request_for(candidates, definitions, include_sheet=True):
    classes = validate_classes(definitions)
    content = [
        {"type": "text", "text": (
            "Classify each numbered visible region by its dominant functional role. "
            "The first image is an unpainted RGB crop; the second is the same crop with numbered masks. "
            "Colors and numbers are annotations, not physical appearance. "
            "Use the RGB to judge the surface inside each numbered mask. "
            + definitions["decision_rules"]
        )},
        {"type": "image_url", "image_url": {"url": "foreground/rgb_crop.png"}},
        {"type": "image_url", "image_url": {"url": "regions/numbered_crop.png"}},
    ]
    if include_sheet:
        content.extend([
            {"type": "text", "text": "Each additional panel highlights one candidate on the same RGB crop."},
            {"type": "image_url", "image_url": {"url": "regions/region_sheet.jpg"}},
        ])
    questions = {
        f"region_{r['id']:02}": {
            "type": "choice",
            "instructions": f"For region {r['id']}, select exactly one dominant role using the supplied definitions. Judge the marked pixels. Use none when no role fits.",
            "criteria": {c["name"]: c["definition"] for c in classes},
        }
        for r in candidates
    }
    return {"state": {"messages": [{"role": "user", "content": content}]}, "questions": questions}


def decode(response, candidates, definitions):
    names = [c["name"] for c in validate_classes(definitions)]
    answers = response.get("answers", {})
    if set(answers) != {f"region_{r['id']:02}" for r in candidates}:
        raise ValueError("Missing or unexpected region IDs in model response")
    results = []
    for region in candidates:
        answer = answers[f"region_{region['id']:02}"]
        probabilities = answer.get("probabilities", {})
        if answer.get("type") != "choice" or set(probabilities) != set(names):
            raise ValueError("Class probabilities do not match the configured ontology")
        values = list(probabilities.values())
        if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in values):
            raise ValueError("Invalid class probability")
        if not math.isclose(sum(values), 1, abs_tol=1e-5):
            raise ValueError("Class probabilities must sum to one")
        choice = answer.get("choice")
        if choice not in probabilities or probabilities[choice] < max(values) - 1e-6:
            raise ValueError("Returned choice is inconsistent with probabilities")
        results.append({
            "region_id": region["id"], "model_choice": choice, "final_choice": choice,
            "probabilities": probabilities, "top_probability": probabilities[choice],
            "postprocess_reason": None,
        })
    return results


def merge(regions, decisions, foreground, definitions, whole_fraction=None):
    classes = validate_classes(definitions)
    if len(regions) != len(decisions):
        raise ValueError("Exactly one decision per region is required")
    coverage = np.zeros(foreground.shape, np.uint16)
    for mask in regions:
        if mask.shape != foreground.shape or (mask & ~foreground).any():
            raise ValueError("Region outside predicted foreground or incorrect dimensions")
        coverage += mask
    if coverage.max() > 1:
        raise ValueError("Candidate regions must not overlap")
    if whole_fraction is not None:
        if not 0 < whole_fraction <= 1:
            raise ValueError("Invalid whole-object threshold")
        for cls in classes:
            if cls["name"] == "none":
                continue
            indices = [i for i, d in enumerate(decisions) if d["final_choice"] == cls["name"]]
            pixels = sum(int(regions[i].sum()) for i in indices)
            if pixels and pixels / max(1, int(foreground.sum())) >= whole_fraction:
                for i in indices:
                    decisions[i]["final_choice"] = "none"
                    decisions[i]["postprocess_reason"] = "whole_object_union_rejected"
    merged = {cls["name"]: np.zeros(foreground.shape, bool) for cls in classes}
    class_map = np.zeros(foreground.shape, np.uint8)
    ids = {cls["name"]: cls["id"] for cls in classes}
    for mask, decision in zip(regions, decisions):
        name = decision["final_choice"]
        merged[name] |= mask
        class_map[mask] = ids[name]
    return merged, class_map
