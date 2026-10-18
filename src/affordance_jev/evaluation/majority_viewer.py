"""Export a portable, offline viewer for the fixed-mask GT-majority diagnostic.

This CPU-only exporter reproduces the saved diagnostic exactly. It never calls
models, rewrites source annotations, or selects independent per-class unions.
"""
import argparse
import base64
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt

from ..common import padded_bbox, read, read_jsonl, sha, write, write_jsonl
from ..datasets.umd import CLASS_NAMES, load_annotation
from .metrics import confusion_matrix, scores_from_confusion


REGION_COLORS = [[0, 0, 0], [115, 68, 226], [35, 170, 80], [225, 75, 58],
                 [32, 151, 207], [217, 145, 29], [195, 62, 170]]


def resolve_data_file(row, key, dataset_root):
    """Relocate old manifests without guessing another sample's identity."""
    original = Path(row[key])
    if original.is_file():
        return original
    instance, frame = row["sample_id"].split("/")
    suffix = "rgb.jpg" if key == "rgb" else "label.mat"
    name = f"{instance}_{frame}_{suffix}"
    root = Path(dataset_root)
    choices = [root / "tools" / instance / name,
               root / "part-affordance-dataset/tools" / instance / name]
    matches = [p for p in choices if p.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Cannot locate {key} for {row['sample_id']}; use --dataset-root")
    return matches[0]


def overlay_files(labels, colors, destination, prefix):
    rgba = np.zeros((*labels.shape, 4), dtype=np.uint8)
    edges = np.zeros_like(rgba)
    for cid in range(1, len(colors)):
        mask = labels == cid
        rgba[mask, :3] = colors[cid]
        rgba[mask, 3] = 255
        boundary = mask & ~binary_erosion(mask)
        edges[boundary] = [*colors[cid], 255]
    if np.any(labels == 255):
        # GT void is shown separately and is excluded from all numeric metrics.
        rgba[labels == 255] = [100, 100, 100, 255]
    for name, data in [(prefix, rgba), (prefix + "_edges", edges)]:
        Image.fromarray(data).save(destination / f"{name}.png", compress_level=6)


def export_sample(task, predictions, dataset_root, output, class_colors):
    row, reference_image, reference_regions, reference_queries = task
    sid = row["sample_id"]
    if len(Path(sid).parts) != 2 or any(p in {".", ".."} for p in Path(sid).parts):
        raise ValueError(f"Invalid sample ID: {sid}")
    source = Path(predictions) / "samples" / sid
    destination = Path(output) / "samples" / sid
    rgb_path = resolve_data_file(row, "rgb", dataset_root)
    gt_path = resolve_data_file(row, "annotation", dataset_root)
    if sha(rgb_path) != row["rgb_sha256"] or sha(gt_path) != row["annotation_sha256"]:
        raise ValueError(f"Source RGB/GT hash changed: {sid}")
    if sha(source / "result/class_map.png") != reference_image["prediction_sha256"]:
        raise ValueError(f"Original prediction changed: {sid}")
    if read(source / "run.json")["status"] != "complete":
        raise ValueError(f"Incomplete source prediction: {sid}")

    gt, kind = load_annotation(gt_path, tuple(row["size_wh"][::-1]))
    if kind != "manual":
        raise ValueError(f"Expected manual GT: {sid}")
    labels = np.asarray(Image.open(source / "regions/label_map.png"))
    original = np.asarray(Image.open(source / "result/class_map.png"))
    foreground = np.asarray(Image.open(source / "foreground/object_mask.png")) > 0
    if not (gt.shape == labels.shape == original.shape == foreground.shape):
        raise ValueError(f"Mismatched pixel grids: {sid}")
    candidates = read(source / "candidates.json")["candidates"]
    decisions = read(source / "decisions.json")["regions"]
    n = len(candidates)
    if n > 6 or len(decisions) != n or len(reference_regions) != n:
        raise ValueError(f"Unexpected region count: {sid}")
    if [c["id"] for c in candidates] != list(range(1, n + 1)):
        raise ValueError(f"Nonsequential candidate IDs: {sid}")
    if labels.max() > n or np.any((labels > 0) & ~foreground):
        raise ValueError(f"Invalid fixed candidate partition: {sid}")
    valid = gt >= 0
    counts = np.bincount(labels[valid].astype(np.int64) * 8 + gt[valid],
                         minlength=(n + 1) * 8).reshape(n + 1, 8)
    predicted_ids = np.zeros(n + 1, dtype=np.uint8)
    names = {v: k for k, v in CLASS_NAMES.items()}
    names["none"] = 0
    for decision in decisions:
        predicted_ids[decision["region_id"]] = names[decision["final_choice"]]
    if not np.array_equal(predicted_ids[labels], original):
        raise AssertionError(f"Original class map and decisions disagree: {sid}")

    majority_ids = np.zeros(n + 1, dtype=np.uint8)
    regions = []
    for index, (candidate, decision, reference) in enumerate(zip(candidates, decisions, reference_regions), 1):
        if reference["region_id"] != index or decision["region_id"] != index:
            raise ValueError(f"Unexpected decision order: {sid}")
        mask = labels == index
        mask_path = source / candidate["mask"]
        if (int(mask.sum()) != candidate["area"]
                or sha(mask_path) != candidate["mask_sha256"]
                or not np.array_equal(mask, np.asarray(Image.open(mask_path)) > 0)):
            raise AssertionError(f"Candidate mask changed: {sid}/{index}")
        count = counts[index]
        # Exactly the saved diagnose.py tie-break rule, including all-void regions.
        target = int(predicted_ids[index]) if count[predicted_ids[index]] == count.max() else int(count.argmax())
        purity = float(count.max() / count.sum()) if count.sum() else None
        if (reference["gt_pixel_counts"] != count.tolist()
                or reference["valid_pixels"] != int(count.sum())
                or reference["gt_majority_id"] != target
                or reference["purity"] != purity
                or reference["prediction"] != decision["final_choice"]):
            raise AssertionError(f"Saved region diagnostic not reproduced: {sid}/{index}")
        majority_ids[index] = target
        x0, y0, x1, y1 = candidate["bbox_xyxy"]
        local = mask[y0:y1, x0:x1]
        distances = distance_transform_edt(np.pad(local, 1))[1:-1, 1:-1]
        y, x = np.unravel_index(distances.argmax(), distances.shape)
        regions.append({
            "id": index, "area": candidate["area"], "valid_pixels": int(count.sum()),
            "gt_counts": count.tolist(), "majority_id": target,
            "majority_class": CLASS_NAMES[target] if target else "none",
            "purity": purity, "original_class": decision["final_choice"],
            "bbox_xyxy": candidate["bbox_xyxy"], "anchor_xy": [int(x + x0), int(y + y0)],
        })
    if majority_ids.tolist() != reference_image["majority_labels"]:
        raise AssertionError(f"Saved majority assignments not reproduced: {sid}")
    majority = majority_ids[labels]
    matrix = confusion_matrix(gt, majority)
    actual_matrix = confusion_matrix(gt, original)
    scores = scores_from_confusion(matrix)
    queries = []
    for reference in reference_queries:
        cid = reference["class_id"]
        iou = scores["per_class"][CLASS_NAMES[cid]]["iou"]
        if iou != reference["majority_iou"]:
            raise AssertionError(f"Saved query IoU not reproduced: {sid}/{cid}")
        queries.append({"class_id": cid, "class": CLASS_NAMES[cid], "iou": iou})
    if len(queries) != int(np.sum(counts.sum(0)[1:] > 0)):
        raise AssertionError(f"Missing GT-present query: {sid}")

    destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(rgb_path, destination / "rgb.jpg")  # Original encoded RGB bytes.
    shutil.copyfile(source / "regions/label_map.png", destination / "candidate_ids.png")
    shutil.copyfile(source / "foreground/object_mask.png", destination / "object_mask.png")
    Image.fromarray(majority).save(destination / "class_map.png")
    gt_png = np.where(gt < 0, 255, gt).astype(np.uint8)
    Image.fromarray(gt_png).save(destination / "gt_class_map.png")
    overlay_files(labels, REGION_COLORS, destination, "candidates")
    overlay_files(majority, class_colors, destination, "majority")
    overlay_files(gt_png, class_colors, destination, "gt")
    # Inspection crop includes both prediction and GT, so a missed object is visible.
    inspection_mask = (labels > 0) | (gt > 0)
    crop = padded_bbox(inspection_mask, .15) if inspection_mask.any() else [0, 0, gt.shape[1], gt.shape[0]]
    record = {
        "sample_id": sid, "object_instance": row["object_instance"],
        "object_category": reference_image["object_category"],
        "size_wh": row["size_wh"], "crop_xyxy": crop,
        "source_crop_xyxy": read(source / "foreground/metadata.json")["crop_xyxy"],
        "path": f"samples/{sid}/", "regions": regions, "queries": queries,
        "mean_present_iou": float(np.mean([q["iou"] for q in queries])),
        "void_pixels": int((gt < 0).sum()),
        "unassigned_gt_foreground_pixels": int(((labels == 0) & (gt > 0)).sum()),
        "background_majority_regions": [r["id"] for r in regions if r["majority_id"] == 0],
        "source_hashes": {"rgb": row["rgb_sha256"], "annotation": row["annotation_sha256"],
                          "original_prediction": reference_image["prediction_sha256"],
                          "candidate_ids": sha(source / "regions/label_map.png")},
        "class_map_sha256": sha(destination / "class_map.png"),
        # file:// browsers may ignore download= on local file links. A data URL
        # makes the original encoded PNG downloadable without fetch/CORS flags.
        "class_map_data_url": "data:image/png;base64," + base64.b64encode(
            (destination / "class_map.png").read_bytes()).decode("ascii"),
    }
    write(destination / "regions.json", record)
    # Read back every saved map; verification is for exported files, not just arrays.
    if not np.array_equal(np.asarray(Image.open(destination / "class_map.png")), majority):
        raise AssertionError(f"Saved map does not match reconstructed output: {sid}")
    return record, matrix, actual_matrix


def copy_web(output):
    shutil.copytree(Path(__file__).with_name("majority_web"), output, dirs_exist_ok=True)


def export_viewer(manifest, predictions, analysis, dataset_root, output, workers=4, web_only=False):
    output = Path(output).resolve()
    if web_only:
        if read(output / "status.json")["status"] != "complete":
            raise ValueError("--web-only requires a completed export")
        copy_web(output)
        print(output / "index.html")
        return
    manifest, predictions, analysis = map(lambda p: Path(p).resolve(), [manifest, predictions, analysis])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing export: {output}")
    started = time.monotonic()
    summary = read(analysis / "summary.json")
    source_hashes = {p.name: sha(p) for p in [manifest, analysis / "summary.json", analysis / "per_image.jsonl",
                                            analysis / "per_region.jsonl", analysis / "per_query.jsonl"]}
    if sha(manifest) != summary["manifest_sha256"]:
        raise ValueError("Use the original diagnostic manifest; --dataset-root relocates its RGB/GT paths")
    rows = read_jsonl(manifest)
    images = {r["sample_id"]: r for r in read_jsonl(analysis / "per_image.jsonl")}
    regions, queries = defaultdict(list), defaultdict(list)
    for name, target in [("per_region", regions), ("per_query", queries)]:
        for row in read_jsonl(analysis / f"{name}.jsonl"):
            target[row["sample_id"]].append(row)
    if (len(rows) != summary["images"] or {r["sample_id"] for r in rows} != set(images)
            or set(images) != set(regions) or set(images) != set(queries)):
        raise ValueError("Manifest and diagnostic sample identities disagree")
    definitions = read(predictions / "affordances_snapshot.json")["classes"]
    class_colors = [[0, 0, 0] for _ in range(8)]
    for c in definitions:
        class_colors[c["id"]] = c["color"]
    tasks = [(r, images[r["sample_id"]], regions[r["sample_id"]], queries[r["sample_id"]]) for r in rows]
    output.mkdir(parents=True, exist_ok=True)
    write(output / "status.json", {"status": "running", "samples": len(rows), "completed": 0})
    records = []
    matrix, actual_matrix = np.zeros((8, 8), dtype=np.int64), np.zeros((8, 8), dtype=np.int64)
    export = partial(export_sample, predictions=str(predictions), dataset_root=str(dataset_root),
                     output=str(output), class_colors=class_colors)
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for number, (record, cm, actual) in enumerate(executor.map(export, tasks, chunksize=4), 1):
                records.append(record)
                matrix += cm
                actual_matrix += actual
                if number % 100 == 0 or number == len(rows):
                    elapsed = time.monotonic() - started
                    print(f"[GT-majority viewer] {number}/{len(rows)} elapsed={elapsed:.1f}s", flush=True)
                    write(output / "status.json", {"status": "running", "samples": len(rows), "completed": number})
        reference_cm = summary["gt_majority_relabel_diagnostic"]["confusion_matrix_gt_rows_pred_columns"]
        if not np.array_equal(matrix, reference_cm) or not np.array_equal(actual_matrix, summary["actual"]["confusion_matrix_gt_rows_pred_columns"]):
            raise AssertionError("Full exported confusion matrix differs from saved diagnostic")
        scores = scores_from_confusion(matrix)
        query_ious = [q["iou"] for r in records for q in r["queries"]]
        g_iou = float(np.mean(query_ious))
        if len(query_ious) != summary["query_count"] or g_iou != summary["queries"]["majority_iou"]:
            raise AssertionError("Full exported query metrics differ from saved diagnostic")
        metrics = {**scores, "images": len(records), "gt_present_queries": len(query_ious),
                   "gIoU_present_queries": g_iou, "P_at_50_present_queries": float(np.mean(np.array(query_ious) >= .5))}
        payload = {"samples": records, "metrics": metrics, "classes": [
            {"id": cid, "name": CLASS_NAMES[cid] if cid else "background / none", "color": class_colors[cid]}
            for cid in range(8)], "region_colors": REGION_COLORS}
        (output / "data.js").write_text("window.MAJORITY_DATA = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + ";\n", encoding="utf-8")
        write(output / "metrics.json", metrics)
        write_jsonl(output / "samples.jsonl", records)
        write(output / "audit.json", {
            "diagnostic": "Fixed masks, single GT-majority class per candidate; NOT independent best class unions",
            "gt": "Original UMD manual dominant GT; no ladle relabel; ignore -1; background participates in majority",
            "tie_rule": "Keep original prediction if tied for highest count, otherwise smallest maximum class ID",
            "uncovered_class_id": 0, "exported_gt_void_id": 255,
            "samples": len(records), "regions": sum(len(r["regions"]) for r in records),
            "queries": len(query_ious), "source_files_sha256": source_hashes,
            "every_rgb_gt_sha_verified": True, "every_candidate_mask_sha_and_partition_verified": True,
            "every_region_count_majority_purity_verified": True, "every_query_iou_exact": True,
            "every_saved_class_map_readback_verified": True,
            "full_majority_and_original_confusion_matrices_exact": True,
            "models_called": False, "source_files_modified": False,
            "exporter_sha256": sha(__file__), "elapsed_seconds": time.monotonic() - started,
        })
        copy_web(output)
        (output / "README.md").write_text(
            "# 固定候选＋GT 多数类别：全量可视化\n\n"
            f"直接打开 `index.html`；全部 {len(records):,} 张样本，mIoU **{scores['foreground_mIoU']:.2%}**，gIoU **{g_iou:.2%}**。\n\n"
            "这是借助 GT 的分割诊断，不是模型推理成绩。保留全部候选 mask，每个候选统一改成有效 GT 像素最多的类别（含背景）；并列优先保留原模型类别，其余取最小类别 ID。未覆盖像素为 0，GT 的 -1 不参与计分。\n\n"
            "整个目录可单独复制；无网络、模型、外部数据依赖。RGB 是原始 JPEG 文件，所有结果与原图同尺寸。\n\n"
            "| 文件 | 内容 |\n|---|---|\n"
            "| `samples/<实例>/<帧>/class_map.png` | 重赋类结果，单通道 0–7 |\n"
            "| `gt_class_map.png` | 原始 GT：0–7，void -1 编码为 255 |\n"
            "| `candidate_ids.png` | 固定候选编号；0 为未分配 |\n"
            "| `rgb.jpg` / `object_mask.png` | 原始 RGB / 原预测物体 mask |\n"
            "| `regions.json` | 候选像素计数、多数类、纯度、查询 IoU 与来源哈希 |\n"
            "| `metrics.json` / `audit.json` | 全量指标 / 与原诊断逐项核对记录 |\n\n"
            "mIoU 是全量混淆矩阵上的七功能类别平均；gIoU 是全部 GT 存在的图像＋功能对的 IoU 平均。界面显示的单样本平均仅统计该图 GT 中出现的功能。\n"
            "默认裁剪包含候选与 GT 的并集，仅供检查；不影响评测，未把 GT 用于原模型的物体选择。\n", encoding="utf-8")
        write(output / "status.json", {"status": "complete", "samples": len(records), "completed": len(records),
                                       "mIoU": scores["foreground_mIoU"], "gIoU": g_iou})
        print(f"Complete: mIoU={scores['foreground_mIoU']:.8%}, gIoU={g_iou:.8%}\n{output / 'index.html'}")
    except Exception as exc:
        write(output / "status.json", {"status": "failed", "completed": len(records), "error": str(exc)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--web-only", action="store_true", help="Refresh static UI files of a completed export")
    export_viewer(**vars(parser.parse_args()))
