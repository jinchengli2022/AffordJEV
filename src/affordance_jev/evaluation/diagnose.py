"""Read-only error attribution for cached UMD PartGLEE + Valen predictions.

GT is used only for retrospective diagnostics. No models or inference files are
modified. The exact query oracle is NOT a deployable multiclass prediction.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from ..common import read, read_jsonl, sha, write, write_jsonl
from ..datasets.umd import CLASS_NAMES, load_annotation
from .metrics import confusion_matrix, scores_from_confusion


def best_union(counts, gt_count, class_id):
    """Exact IoU over all subsets of <= 6 disjoint regions, including empty."""
    n = len(counts)
    if n > 6:
        raise ValueError("This diagnostic expects the frozen six-region run")
    bits = (np.arange(1 << n)[:, None] >> np.arange(n)) & 1
    intersection = bits @ counts[:, class_id]
    area = bits @ counts.sum(1)
    scores = intersection / np.maximum(1, gt_count + area - intersection)
    best = int(scores.argmax())
    return float(scores[best]), (np.flatnonzero(bits[best]) + 1).tolist()


def analyze_sample(row, predictions):
    out = Path(predictions) / "samples" / row["sample_id"]
    if sha(row["annotation"]) != row["annotation_sha256"]:
        raise ValueError(f"Changed GT: {row['sample_id']}")
    gt, kind = load_annotation(row["annotation"], tuple(row["size_wh"][::-1]))
    if kind != "manual" or read(out / "run.json")["status"] != "complete":
        raise ValueError("Expected completed inference on manual annotations")
    valid = gt >= 0
    foreground = np.asarray(Image.open(out / "foreground/object_mask.png")) > 0
    labels = np.asarray(Image.open(out / "regions/label_map.png"))
    pred = np.asarray(Image.open(out / "result/class_map.png"))
    if not (gt.shape == foreground.shape == labels.shape == pred.shape):
        raise ValueError("Mismatched pixel grids")
    candidates = read(out / "candidates.json")["candidates"]
    decisions = read(out / "decisions.json")["regions"]
    n = len(candidates)
    if [c["id"] for c in candidates] != list(range(1, n + 1)):
        raise ValueError("Nonsequential candidate IDs")
    if int(labels.max()) > n or np.any((labels > 0) & ~foreground):
        raise ValueError("Invalid candidate partition")
    name_ids = {v: k for k, v in CLASS_NAMES.items()}
    name_ids["none"] = 0
    predicted_ids = np.zeros(n + 1, dtype=np.uint8)
    for d in decisions:
        predicted_ids[d["region_id"]] = name_ids[d["final_choice"]]
    if len(decisions) != n or not np.array_equal(predicted_ids[labels], pred):
        raise ValueError("Saved masks, decisions and prediction disagree")
    counts = np.bincount(
        labels[valid].astype(np.int64) * 8 + gt[valid], minlength=(n + 1) * 8
    ).reshape(n + 1, 8)
    region_counts = counts[1:]
    gt_counts = counts.sum(0)
    actual_cm = confusion_matrix(gt, pred)
    majority_ids = np.zeros(n + 1, dtype=np.uint8)
    majority_cm = np.zeros((8, 8), dtype=np.int64)
    majority_cm[:, 0] += counts[0]
    region_rows = []
    for i, (candidate, decision, count) in enumerate(zip(candidates, decisions, region_counts), 1):
        if decision["region_id"] != i or candidate["area"] != int((labels == i).sum()):
            raise ValueError("Candidate metadata disagreement")
        # Keep the existing prediction if it is tied for the GT majority.
        target = int(predicted_ids[i]) if count[predicted_ids[i]] == count.max() else int(count.argmax())
        majority_ids[i] = target
        majority_cm[:, target] += count
        probabilities = decision["probabilities"]
        ranking = sorted(probabilities, key=probabilities.get, reverse=True)
        target_name = CLASS_NAMES[target] if target else "none"
        region_rows.append({
            "sample_id": row["sample_id"], "region_id": i, "raw_id": candidate["raw_id"],
            "valid_pixels": int(count.sum()), "gt_pixel_counts": count.tolist(),
            "gt_majority_id": target, "gt_majority_class": target_name,
            "purity": float(count.max() / count.sum()) if count.sum() else None,
            "prediction": decision["final_choice"], "correct_majority": int(predicted_ids[i]) == target,
            "target_in_top2": target_name in ranking[:2],
            "target_probability": probabilities[target_name],
            "prediction_probability": probabilities[decision["final_choice"]],
            "top2_margin": probabilities[ranking[0]] - probabilities[ranking[1]],
        })
    fg_gt = (gt > 0) & foreground
    gt_positive = int((gt > 0).sum())
    # A disjoint partition gives an exact, additive pixel-error decomposition.
    error_counts = {
        "foreground_missed_gt": int(((gt > 0) & ~foreground).sum()),
        "partition_unassigned_gt": int((fg_gt & (labels == 0)).sum()),
        "mixed_region_unavoidable": int(sum(c.sum() - c.max() for c in region_counts)),
        "decision_excess_over_majority": int(sum(c.max() - c[predicted_ids[i + 1]] for i, c in enumerate(region_counts))),
    }
    total_errors = int(actual_cm.sum() - np.trace(actual_cm))
    if sum(error_counts.values()) != total_errors:
        raise AssertionError("Pixel-error attribution does not sum to actual errors")
    crop = read(out / "foreground/metadata.json")["crop_xyxy"]
    x0, y0, x1, y1 = crop
    gt_crop = gt[y0:y1, x0:x1]
    valid_crop = gt_crop >= 0
    fg_crop = foreground[y0:y1, x0:x1]
    with np.load(out / "raw/proposals.npz", allow_pickle=False) as raw:
        raw_masks = raw["masks_crop"].astype(bool)
    if raw_masks.shape[1:] != gt_crop.shape:
        raise ValueError("Raw proposal crop does not align with GT")
    clipped = raw_masks & fg_crop
    raw_areas = (raw_masks & valid_crop).sum((1, 2))
    clipped_areas = (clipped & valid_crop).sum((1, 2))
    actual_scores = scores_from_confusion(actual_cm)
    majority_scores = scores_from_confusion(majority_cm)
    queries = []
    for cid in range(1, 8):
        if not gt_counts[cid]:
            continue
        oracle_iou, selected = best_union(region_counts, int(gt_counts[cid]), cid)
        raw_inter = (raw_masks & (gt_crop == cid)).sum((1, 2))
        clipped_inter = (clipped & (gt_crop == cid)).sum((1, 2))
        raw_ious = raw_inter / np.maximum(1, gt_counts[cid] + raw_areas - raw_inter)
        clipped_ious = clipped_inter / np.maximum(1, gt_counts[cid] + clipped_areas - clipped_inter)
        final_single = region_counts[:, cid] / np.maximum(1, gt_counts[cid] + region_counts.sum(1) - region_counts[:, cid])
        actual = actual_scores["per_class"][CLASS_NAMES[cid]]["iou"]
        fg_recall = float(((gt == cid) & foreground).sum() / gt_counts[cid])
        if actual > oracle_iou + 1e-12 or oracle_iou > fg_recall + 1e-12:
            raise AssertionError("Query bound inconsistency")
        queries.append({
            "sample_id": row["sample_id"], "class_id": cid, "class": CLASS_NAMES[cid],
            "gt_pixels": int(gt_counts[cid]), "actual_iou": actual,
            "majority_iou": majority_scores["per_class"][CLASS_NAMES[cid]]["iou"],
            "foreground_recall_upper_bound": fg_recall,
            "candidate_coverage_recall": float(region_counts[:, cid].sum() / gt_counts[cid]),
            "best_union_iou": oracle_iou, "best_union_region_ids": selected,
            "best_single_final_iou": float(final_single.max()) if n else 0.0,
            "best_single_raw_iou": float(raw_ious.max()), "best_single_raw_id": int(raw_ious.argmax()),
            "best_single_clipped_iou": float(clipped_ious.max()), "best_single_clipped_id": int(clipped_ious.argmax()),
        })
    trace = read(out / "regions/filter_trace.json")
    bg_fp = (gt == 0) & (pred > 0)
    distance = distance_transform_edt(gt <= 0)
    image_row = {
        "sample_id": row["sample_id"], "object_category": row["object_instance"].rsplit("_", 1)[0],
        "prediction_sha256": sha(out / "result/class_map.png"),
        "region_count": n, "gt_foreground_pixels": gt_positive,
        "foreground_gt_intersection": int(fg_gt.sum()),
        "foreground_valid_pixels": int((foreground & valid).sum()),
        "foreground_recall": float(fg_gt.sum() / gt_positive),
        "foreground_precision_against_annotated_pixels": float(fg_gt.sum() / max(1, (foreground & valid).sum())),
        "background_false_positive_pixels": int(bg_fp.sum()),
        "background_false_positive_within_3px": int((bg_fp & (distance <= 3)).sum()),
        "background_false_positive_beyond_10px": int((bg_fp & (distance > 10)).sum()),
        "pixel_errors": total_errors, "pixel_error_decomposition": error_counts,
        "cap_was_applied": any(a["reason"] == "region_cap" for a in trace["assignment"]),
        "raw_filter_rejections": dict(Counter(r["rejection"] or "accepted" for r in trace["filter"])),
        "mean_actual_iou": float(np.mean([q["actual_iou"] for q in queries])),
        "mean_best_union_iou": float(np.mean([q["best_union_iou"] for q in queries])),
        "majority_labels": majority_ids.tolist(),
    }
    return image_row, queries, region_rows, actual_cm, majority_cm


def query_summary(rows):
    keys = ["actual_iou", "majority_iou", "foreground_recall_upper_bound", "candidate_coverage_recall",
            "best_union_iou", "best_single_final_iou", "best_single_raw_iou", "best_single_clipped_iou"]
    return {"queries": len(rows), **{k: float(np.mean([r[k] for r in rows])) for k in keys},
            "actual_P50": float(np.mean([r["actual_iou"] >= .5 for r in rows])),
            "best_union_P50": float(np.mean([r["best_union_iou"] >= .5 for r in rows]))}


def region_summary(rows):
    if not rows:
        return {"regions": 0}
    return {"regions": len(rows), "agreement_with_gt_majority": float(np.mean([r["correct_majority"] for r in rows])),
            "target_in_top2_rate": float(np.mean([r["target_in_top2"] for r in rows])),
            "mean_purity": float(np.mean([r["purity"] for r in rows]))}


def diagnose(manifest, predictions, output, workers=4, limit=None):
    started = time.monotonic()
    manifest, predictions, output = Path(manifest).resolve(), Path(predictions).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty: {output}")
    manifest_hash = sha(manifest)
    reference = read(predictions / "evaluation/metrics.json")
    if manifest_hash != reference["manifest_sha256"]:
        raise ValueError("Use the original evaluation manifest")
    rows = read_jsonl(manifest)
    if limit is not None:
        rows = rows[:limit]
    output.mkdir(parents=True, exist_ok=True)
    images, queries, regions = [], [], []
    actual_cm = np.zeros((8, 8), dtype=np.int64)
    majority_cm = np.zeros((8, 8), dtype=np.int64)
    write(output / "status.json", {"status": "running", "samples": len(rows)})
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for i, (im, qs, rs, a, m) in enumerate(executor.map(partial(analyze_sample, predictions=str(predictions)), rows, chunksize=8), 1):
            images.append(im)
            queries.extend(qs)
            regions.extend(rs)
            actual_cm += a
            majority_cm += m
            if i % 100 == 0 or i == len(rows):
                print(f"[diagnose] {i}/{len(rows)} elapsed={time.monotonic() - started:.1f}s", flush=True)
                write(output / "status.json", {"status": "running", "completed": i, "samples": len(rows)})
    if limit is None and not np.array_equal(actual_cm, reference["metrics"]["confusion_matrix_gt_rows_pred_columns"]):
        raise AssertionError("Cached predictions do not reproduce original metrics")
    if sha(manifest) != manifest_hash:
        raise ValueError("Manifest changed during analysis")
    categories = sorted({r["object_category"] for r in images})
    category_cm = {cat: np.zeros((8, 8), dtype=np.int64) for cat in categories}
    # Saved baseline already contains per-instance matrices; avoid rereading images.
    for instance, metrics in reference["per_object"].items():
        cat = instance.rsplit("_", 1)[0]
        if cat in category_cm:
            category_cm[cat] += metrics["confusion_matrix_gt_rows_pred_columns"]
    counts = Counter()
    for row in images:
        counts.update(row["pixel_error_decomposition"])
    majority_metrics = scores_from_confusion(majority_cm)
    summary = {
        "manifest": str(manifest), "manifest_sha256": manifest_hash,
        "predictions": str(predictions), "source_metrics_sha256": sha(predictions / "evaluation/metrics.json"),
        "script_sha256": sha(__file__), "images": len(images), "query_count": len(queries),
        "baseline_exactly_reproduced": limit is None,
        "actual": scores_from_confusion(actual_cm), "gt_majority_relabel_diagnostic": majority_metrics,
        "queries": query_summary(queries),
        "per_function": {CLASS_NAMES[c]: query_summary([q for q in queries if q["class_id"] == c]) for c in range(1, 8) if any(q["class_id"] == c for q in queries)},
        "per_object_category": {cat: query_summary([q for q in queries if q["sample_id"].split('/')[0].rsplit('_', 1)[0] == cat]) for cat in categories},
        "pixel_error_decomposition": dict(counts), "total_pixel_errors": sum(counts.values()),
        "regions": region_summary(regions),
        "high_purity_foreground_regions": region_summary([r for r in regions if r["gt_majority_id"] > 0 and r["purity"] is not None and r["purity"] >= .9]),
        "region_count_histogram": dict(Counter(r["region_count"] for r in images)),
        "images_with_region_cap_applied": sum(r["cap_was_applied"] for r in images),
        "foreground_pixel_recall": sum(r["foreground_gt_intersection"] for r in images) / sum(r["gt_foreground_pixels"] for r in images),
        "foreground_precision_against_annotated_pixels": sum(r["foreground_gt_intersection"] for r in images) / sum(r["foreground_valid_pixels"] for r in images),
        "foreground_recall_below_50_images": sum(r["foreground_recall"] < .5 for r in images),
        "background_false_positive_pixels": sum(r["background_false_positive_pixels"] for r in images),
        "background_false_positive_within_3px": sum(r["background_false_positive_within_3px"] for r in images),
        "background_false_positive_beyond_10px": sum(r["background_false_positive_beyond_10px"] for r in images),
        "protocol": {
            "gt": "Original UMD manual dominant labels; no ladle relabel; background false positives count; void -1 ignored.",
            "majority": "GT majority per fixed candidate maximizes correctly labeled pixels, NOT mIoU. Background majority becomes none. Diagnostic only.",
            "query_oracle": "Enumerate all 2^R unions of fixed disjoint candidates, independently for each GT-present function. Upper bound on present-query gIoU; choices may conflict across functions, so not a realizable multiclass result.",
            "raw": "Best SINGLE raw / foreground-clipped proposal selected with GT; not a union upper bound.",
            "error_decomposition": "Additive counts of incorrectly labeled pixels, including GT background; percentages are NOT additive IoU losses.",
            "background": "GT background may include unlabeled physical surfaces or boundary disagreement; it does not always mean unrelated scenery.",
            "test_usage": "Retrospective diagnosis only. No training, model inference, parameter selection or changes to source predictions.",
        },
    }
    if limit is None:
        summary["per_object_category_actual"] = {cat: scores_from_confusion(cm) for cat, cm in category_cm.items()}
    write(output / "summary.json", summary)
    write_jsonl(output / "per_image.jsonl", images)
    write_jsonl(output / "per_query.jsonl", queries)
    write_jsonl(output / "per_region.jsonl", regions)
    write(output / "status.json", {"status": "complete", "samples": len(images), "queries": len(queries),
                                   "elapsed_seconds": time.monotonic() - started, "model_inference_calls": 0})
    print(f"Complete: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    diagnose(**vars(args))


if __name__ == "__main__":
    main()
