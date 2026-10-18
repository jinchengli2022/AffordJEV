"""Rebuild the fixed 76-image UMD training selection from original UMD files.

The selection and source checksums are versioned in configs/umd_train_76.json.
No OOAL import, author archive, model weight or historical output is required.
"""
import argparse
from collections import Counter
from pathlib import Path

from PIL import Image

from ..common import read, read_jsonl, sha, write, write_jsonl
from .umd import load_annotation, read_split, resolve_root


def _category(instances):
    return {name.rsplit("_", 1)[0] for name in instances}


def _source(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Selection paths must be relative to the UMD root: {relative}")
    source = (root / relative).resolve(strict=True)
    if not source.is_relative_to(root):
        raise ValueError(f"Selection source leaves the UMD root: {source}")
    return source


def prepare(dataset_dir, selection_config, test_manifest, output, verify_only=False):
    root = resolve_root(dataset_dir, "tools")
    selection_path = Path(selection_config).resolve(strict=True)
    selection = read(selection_path)
    output = Path(output).resolve()
    if not verify_only and (output.exists() or output.with_suffix(".meta.json").exists()):
        raise FileExistsError(f"Existing manifest is protected: {output}; use --verify-only to check it")

    split_path = root / "novel_split.txt"
    if sha(split_path) != selection["novel_split_sha256"]:
        raise ValueError("The official novel_split.txt checksum differs from the pinned protocol")
    split = read_split(split_path)
    expected_train = {name for name, role in split.items() if role == "train"}
    expected_test = {name for name, role in split.items() if role == "test"}
    rows = []
    for fixed in selection["samples"]:
        row = dict(fixed)
        instance = row["object_instance"]
        if instance not in expected_train or row["split"] != "novel/train":
            raise ValueError(f"Non-training instance in the fixed selection: {instance}")
        if row["sample_id"] != f"{instance}/{row['frame_index']:08d}":
            raise ValueError(f"Inconsistent training sample identity: {row['sample_id']}")
        for key in ("rgb", "annotation"):
            source = _source(root, row[key])
            if sha(source) != row[f"{key}_sha256"]:
                raise ValueError(f"Original {key} checksum mismatch: {row['sample_id']}")
            row[key] = str(source)
        with Image.open(row["rgb"]) as image:
            size = list(image.size)
        if size != row["size_wh"]:
            raise ValueError(f"Original RGB dimensions changed: {row['sample_id']}")
        _, kind = load_annotation(row["annotation"], tuple(size[::-1]))
        if kind != row["gt_type"]:
            raise ValueError(f"Annotation provenance changed: {row['sample_id']}")
        rows.append(row)

    instances = {r["object_instance"] for r in rows}
    if len(rows) != 76 or len(instances) != 76 or instances != expected_train:
        raise ValueError("Expected exactly one fixed image for each of the 76 novel/train instances")
    types = dict(Counter(r["gt_type"] for r in rows))
    if types != {"manual": 32, "automatic": 44} or types != selection["training_annotation_types"]:
        raise ValueError(f"Training annotation mix changed: {types}")
    if _category(instances) != set(selection["training_categories"]):
        raise ValueError("Training categories differ from the pinned selection")

    test_path = Path(test_manifest).resolve(strict=True)
    tests = read_jsonl(test_path)
    test_instances = {r["object_instance"] for r in tests}
    if len(tests) != selection["test_samples"] or len({r["sample_id"] for r in tests}) != len(tests):
        raise ValueError("Expected the complete 2661-image UMD manual novel/test manifest, without duplicates")
    if test_instances != expected_test or len(test_instances) != selection["test_instances"]:
        raise ValueError("Test manifest does not cover exactly the 29 official novel/test instances")
    if _category(test_instances) != set(selection["test_categories"]):
        raise ValueError("Test categories differ from the pinned protocol")
    for row in tests:
        if row["gt_type"] != "manual":
            raise ValueError(f"Test manifest includes a non-manual frame: {row['sample_id']}")
        instance, frame = row["sample_id"].split("/")
        if instance != row["object_instance"] or int(frame) != row["frame_index"]:
            raise ValueError(f"Inconsistent test sample identity: {row['sample_id']}")
        for key, suffix in (("rgb", "_rgb.jpg"), ("annotation", "_label.mat")):
            expected = root / "tools" / instance / f"{instance}_{frame}{suffix}"
            if Path(row[key]).resolve(strict=True) != expected.resolve(strict=True):
                raise ValueError(f"Test manifest points outside this dataset: {row[key]}; rebuild it with prepare_umd.sh")
            if sha(expected) != row[f"{key}_sha256"]:
                raise ValueError(f"Test input checksum mismatch: {row['sample_id']} {key}")
    for key in ("sample_id", "object_instance", "rgb_sha256"):
        if {r[key] for r in rows} & {r[key] for r in tests}:
            raise ValueError(f"Training/test overlap: {key}")
    if _category(instances) & _category(test_instances):
        raise ValueError("Training and test object categories overlap")
    if verify_only and output.exists() and read_jsonl(output) != rows:
        raise ValueError(f"Existing manifest differs from the fixed selection: {output}")

    report = {
        "training_samples": len(rows), "training_instances": len(instances),
        "training_categories": sorted(_category(instances)), "annotation_types": types,
        "test_samples": len(tests), "test_instances": len(test_instances),
        "test_categories": sorted(_category(test_instances)),
        "instance_overlap": [], "category_overlap": [], "sample_overlap": [], "rgb_file_hash_overlap": [],
        "selection_config_sha256": sha(selection_path), "split_sha256": sha(split_path),
        "test_manifest_sha256": sha(test_path), "verified_training_rgb_and_mat_checksums": len(rows),
        "verified_test_rgb_and_mat_checksums": len(tests),
        "selection_source_verification": selection["source_verification"],
        "label_policy": selection["label_policy"], "verify_only": verify_only,
    }
    if not verify_only:
        write_jsonl(output, rows)
        report["train_manifest_sha256"] = sha(output)
        write(output.with_suffix(".meta.json"), report)
    return report


def main():
    parser = argparse.ArgumentParser(description="Rebuild/verify the fixed 76-image UMD training selection without OOAL")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--selection-config", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify-only", action="store_true", help="Read-only checks; also compare output when it exists")
    args = parser.parse_args()
    report = prepare(args.dataset_dir, args.selection_config, args.test_manifest, args.output, args.verify_only)
    print(f"Verified {report['training_samples']} training images (32 manual + 44 automatic), "
          f"{report['test_samples']} test images; no sample, instance, category or RGB hash overlap.")
    print("Read-only verification completed" if args.verify_only else f"Wrote {args.output}")


if __name__ == "__main__":
    main()
