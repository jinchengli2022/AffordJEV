import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mask(path, shape=None):
    array = np.asarray(Image.open(path))
    if array.ndim != 2 or not set(np.unique(array)).issubset({0, 1, 255}):
        raise ValueError(f"Expected a single-channel binary mask: {path}")
    if shape is not None and array.shape != tuple(shape):
        raise ValueError(f"Mask/RGB dimensions disagree: {path}: {array.shape} != {shape}")
    return array > 0


def save_mask(path, mask):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(path)


def bbox(mask):
    ys, xs = np.where(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None


def padded_bbox(mask, fraction):
    box = bbox(mask)
    if box is None:
        raise ValueError("Object foreground is empty")
    x0, y0, x1, y1 = box
    padding = max(1, round(max(x1 - x0, y1 - y0) * fraction))
    h, w = mask.shape
    return [max(0, x0-padding), max(0, y0-padding), min(w, x1+padding), min(h, y1+padding)]


def validate_classes(config):
    classes = config["classes"]
    names = [x["name"] for x in classes]
    ids = [x["id"] for x in classes]
    if config.get("mode") != "single_label" or len(set(names)) != len(names):
        raise ValueError("Supply unique mutually exclusive functional classes")
    if len(set(ids)) != len(ids) or any(type(i) is not int or not 0 <= i <= 255 for i in ids):
        raise ValueError("Class IDs must be unique uint8 integers")
    if config.get("unassigned_id") != 0 or not any(c["name"] == "none" and c["id"] == 0 for c in classes):
        raise ValueError("Use 0 for unmatched/background; preserve the explicit none mask separately")
    if not config.get("decision_rules") or any(not x.get("definition") for x in classes):
        raise ValueError("Provide explicit functional definitions and single-label rules")
    return classes


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{number}") from exc
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)
