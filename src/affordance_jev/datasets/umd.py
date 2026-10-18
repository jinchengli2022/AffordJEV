"""Read original UMD files; ground truth is used only in indexing/evaluation."""
import argparse
from collections import Counter
from pathlib import Path
import re

import numpy as np
from PIL import Image
from scipy.io import loadmat

from ..common import sha, write, write_jsonl


CLASS_NAMES = {
    0: "background", 1: "grasp", 2: "cut", 3: "scoop", 4: "contain",
    5: "pound", 6: "support", 7: "wrap-grasp",
}
SOURCE = "https://users.umiacs.umd.edu/~fermulcm/affordance/part-affordance-dataset/README.txt"


def annotation_type(data):
    value = data.get("gt_type")
    if value is None:
        raise ValueError("Missing gt_type: cannot verify manual annotation provenance")
    flat = np.asarray(value).reshape(-1)
    if flat.size != 1:
        raise ValueError("Expected a single gt_type string")
    value = flat[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    value = str(value).strip().lower()
    if value not in {"manual", "automatic"}:
        raise ValueError(f"Unknown gt_type: {value!r}")
    return value


def load_annotation(path, expected_shape=None):
    data = loadmat(str(path), variable_names=["gt_label", "gt_type"])
    if "gt_label" not in data:
        raise ValueError(f"Missing gt_label: {path}")
    label = np.asarray(data["gt_label"])
    if label.ndim != 2 or not np.issubdtype(label.dtype, np.integer):
        raise ValueError(f"Expected a 2D integer dominant-label map: {path}")
    if expected_shape is not None and label.shape != tuple(expected_shape):
        raise ValueError(f"RGB/GT dimension mismatch: {path}: {label.shape} != {expected_shape}")
    values = set(np.unique(label).tolist())
    if not values <= set(CLASS_NAMES) | {-1}:
        raise ValueError(f"Unknown UMD labels {values - set(CLASS_NAMES) - {-1}} in {path}")
    # -1 occurs in the original clutter release. It is void, never background.
    return label.astype(np.int16), annotation_type(data)


def read_split(path, mode="novel", fold=1):
    """Return instance -> train/test using the original split file."""
    if mode == "fold" and not 1 <= fold <= 10:
        raise ValueError("fold must be 1..10")
    result = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "%")):
            continue
        if mode == "fold":
            match = re.fullmatch(r"folds:\s*([01\s]+)\s+object:\s*(\S+)", line)
            if not match:
                raise ValueError(f"Invalid official fold line: {line}")
            flags = [int(v) for v in match.group(1).split()]
            if len(flags) != 10:
                raise ValueError(f"Expected 10 fold columns: {line}")
            name, role = match.group(2), "test" if flags[fold - 1] == 1 else "train"
        else:
            fields = line.split()
            if len(fields) != 2 or fields[0] not in {"1", "2"}:
                raise ValueError(f"Invalid official split line: {line}")
            name, role = fields[1], "train" if fields[0] == "1" else "test"
        if name in result:
            raise ValueError(f"Duplicate instance in split: {name}")
        if "/" in name or "\\" in name or name in {".", ".."}:
            raise ValueError(f"Invalid instance name: {name}")
        result[name] = role
    if not result:
        raise ValueError(f"Empty split: {path}")
    return result


def resolve_root(path, subset):
    path = Path(path).resolve(strict=True)
    if (path / subset).is_dir():
        return path
    if path.name == subset and path.is_dir():
        return path.parent
    nested = path / ("part-affordance-dataset" if subset == "tools" else "part-affordance-clutter")
    if (nested / subset).is_dir():
        return nested
    raise FileNotFoundError(
        f"No {subset}/ under {path}. Extract the completed official archive first; "
        "an incomplete download is not a dataset."
    )


def build_manifest(root, output, split="novel", partition="test", fold=1,
                   subset="tools", allow_incomplete=False, limit=0):
    root = resolve_root(root, subset)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Manifest already exists: {output}")
    if partition not in {"train", "test", "all"}:
        raise ValueError("partition must be train/test/all")
    if split == "all" and partition != "all":
        raise ValueError("Use partition=all with split=all")
    if subset == "clutter" and (split != "all" or partition != "all"):
        raise ValueError("Clutter has no tool-instance split; use split=all, partition=all")
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    directory = root / subset
    available = {p.name for p in directory.iterdir() if p.is_dir()}
    split_path = None
    if split == "all":
        selected = sorted(available)
        missing = []
        if subset == "tools":
            roster = root / "novel_split.txt"
            if roster.exists():
                missing = sorted(set(read_split(roster)) - available)
            elif not allow_incomplete:
                raise FileNotFoundError("Need novel_split.txt to check the complete tools roster")
    else:
        filenames = {"novel": "novel_split.txt", "category": "category_split.txt", "fold": "category_10_fold.txt"}
        split_path = root / filenames[split]
        assignments = read_split(split_path, split, fold)
        expected = {name for name, role in assignments.items() if partition == "all" or role == partition}
        missing = sorted(expected - available)
        selected = sorted(expected & available)
    if missing and not allow_incomplete:
        raise FileNotFoundError(f"Selected split is incomplete: {len(missing)} instances missing, e.g. {missing[:6]}")
    if not selected:
        raise ValueError("No selected object directories")
    rows, types, counts = [], Counter(), Counter()
    for number, instance in enumerate(selected, 1):
        images = sorted((directory / instance).glob("*_rgb.jpg"))
        if not images:
            raise ValueError(f"No RGB images in {directory / instance}")
        for image in images:
            match = re.fullmatch(re.escape(instance) + r"_(\d+)_rgb\.jpg", image.name)
            if not match:
                raise ValueError(f"Unexpected UMD filename: {image.name}")
            frame = int(match.group(1))
            label_path = image.with_name(image.name.replace("_rgb.jpg", "_label.mat"))
            rank_path = image.with_name(image.name.replace("_rgb.jpg", "_label_rank.mat"))
            with Image.open(image) as im:
                width, height = im.size
            label, kind = load_annotation(label_path, (height, width))
            types[kind] += 1
            if kind != "manual":
                continue
            rows.append({
                "sample_id": f"{instance}/{frame:08d}", "rgb": str(image),
                "annotation": str(label_path), "rank_annotation": str(rank_path) if rank_path.is_file() else None,
                "object_instance": instance, "frame_index": frame, "gt_type": kind,
                "size_wh": [width, height], "rgb_sha256": sha(image),
                "annotation_sha256": sha(label_path), "void_pixels": int((label == -1).sum()),
            })
            counts[instance] += 1
        print(f"[UMD index] {number}/{len(selected)} {instance}: {counts[instance]} manual frames", flush=True)
    if not rows:
        raise ValueError("No verified manual frames selected")
    full_count = len(rows)
    if limit:
        rows = rows[:limit]
    write_jsonl(output, rows)
    meta = {
        "dataset": "UMD Part Affordance", "source": SOURCE, "dataset_root": str(root),
        "subset": subset, "split": split, "partition": partition,
        "fold": fold if split == "fold" else None,
        "split_file": str(split_path) if split_path else None,
        "split_sha256": sha(split_path) if split_path else None,
        "manifest_sha256": sha(output), "samples": len(rows),
        "manual_frames_before_limit": full_count, "limit": limit,
        "annotation_types_scanned": dict(types),
        "selected_instances": selected, "manual_frames_per_instance_before_limit": dict(counts),
        "missing_instances": missing, "allow_incomplete": allow_incomplete,
        "complete_selected_split": not missing and not limit and not allow_incomplete,
        "inference_inputs": ["RGB", "global affordance definitions"],
        "frame_selection": "gt_type == manual; not filename modulo or dtype",
        "label_mode": "dominant gt_label, not ranked multi-label annotations",
        "class_ids": CLASS_NAMES, "ignore_gt_ids": [-1],
    }
    write(output.with_suffix(".meta.json"), meta)
    return meta


def main():
    parser = argparse.ArgumentParser(description="Index original UMD manual frames without copying or resizing images.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subset", choices=["tools", "clutter"], default="tools")
    parser.add_argument("--split", choices=["novel", "category", "fold", "all"], default="novel")
    parser.add_argument("--partition", choices=["train", "test", "all"], default="test")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-incomplete", action="store_true", help="Development only; recorded in metadata.")
    args = parser.parse_args()
    meta = build_manifest(args.dataset_dir, args.output, args.split, args.partition,
                          args.fold, args.subset, args.allow_incomplete, args.limit)
    print(f"Indexed {meta['samples']} manual RGB frames -> {args.output}")


if __name__ == "__main__":
    main()
