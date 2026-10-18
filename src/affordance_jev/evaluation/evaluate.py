"""Evaluate saved predictions only. GT never feeds either inference model."""
import argparse
from collections import Counter
import csv
import html
from pathlib import Path

import numpy as np
from PIL import Image

from ..common import read, write, read_jsonl, write_jsonl, sha
from ..datasets.umd import load_annotation
from ..pipeline.run import sample_directory
from ..visualization import overlay, sheet
from .metrics import Metrics


def evaluate(manifest_path, predictions, output, vis_limit=50, allow_missing=False):
    manifest_path = Path(manifest_path).resolve(strict=True)
    predictions = Path(predictions).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Evaluation output must be empty: {output}")
    if vis_limit < 0:
        raise ValueError("vis_limit must be nonnegative; 0 disables visualizations")
    run = read(predictions / "run.json")
    if run.get("status") == "running":
        raise ValueError("Inference is still running")
    if run.get("evaluation_manifest_sha256") != sha(manifest_path):
        raise ValueError("Use the exact evaluation_manifest.jsonl saved by this inference run")
    rows = read_jsonl(manifest_path)
    if not rows or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Empty or duplicated evaluation samples")
    definitions = read(predictions / "affordances_snapshot.json")
    classes = [c for c in definitions["classes"] if c["id"] != 0]
    output.mkdir(parents=True, exist_ok=True)
    total, per_object = Metrics(), {}
    failures = Counter()
    per_image, queries, cards = [], [], []
    durations = []
    for index, row in enumerate(rows, 1):
        sample = sample_directory(predictions, row["sample_id"])
        width, height = row["size_wh"]
        if sha(row["annotation"]) != row["annotation_sha256"]:
            raise ValueError(f"GT changed after indexing: {row['sample_id']}")
        if sha(row["rgb"]) != row["rgb_sha256"]:
            raise ValueError(f"RGB changed after indexing: {row['sample_id']}")
        gt, kind = load_annotation(row["annotation"], (height, width))
        if kind != "manual" or row.get("gt_type") != "manual":
            raise ValueError("Automatic labels must not enter this evaluation")
        state_path = sample / "run.json"
        state = read(state_path) if state_path.is_file() else {"status": "missing"}
        prediction_path = sample / "result/class_map.png"
        failure = None
        if state["status"] == "complete" and prediction_path.is_file():
            prediction = np.asarray(Image.open(prediction_path))
            prediction_sha = sha(prediction_path)
        elif state["status"] == "failed":
            failure = "recorded_inference_failure"
            prediction = np.zeros(gt.shape, np.uint8)
            prediction_sha = None
        elif allow_missing:
            failure = "missing_or_unfinished_prediction"
            prediction = np.zeros(gt.shape, np.uint8)
            prediction_sha = None
        else:
            raise FileNotFoundError(f"Missing completed prediction: {row['sample_id']}. Use --allow-missing to count as zero.")
        if failure:
            failures[failure] += 1
        image_queries = total.add(gt, prediction)
        object_metrics = per_object.setdefault(row["object_instance"], Metrics())
        object_metrics.add(gt, prediction)
        per_image.append({
            "sample_id": row["sample_id"], "failure": failure, "prediction_sha256": prediction_sha,
            "ignored_gt_pixels": int((gt == -1).sum()),
            "mean_IoU_present_classes": float(np.mean([r["iou"] for r in image_queries])) if image_queries else None,
            "queries": image_queries,
        })
        queries.extend({"sample_id": row["sample_id"], "failure": failure, **q} for q in image_queries)
        if failure is None and (sample / "timings.json").is_file():
            timing = read(sample / "timings.json")
            part = timing["partglee"]["timings"]
            durations.append({
                "partglee_two_forward_seconds": part["object_forward_seconds"] + part["part_forward_seconds"],
                "valen_predict_seconds": timing["valen"]["timings"]["predict_seconds"],
                "valen_first_model_request": timing["valen"]["first_model_request"],
            })
        if index <= vis_limit:
            rgb = Image.open(row["rgb"]).convert("RGB")
            gt_im = overlay(rgb, [gt == c["id"] for c in classes],
                            colors=[c["color"] for c in classes], labels=[c["name"] for c in classes])
            pred_im = overlay(rgb, [prediction == c["id"] for c in classes],
                              colors=[c["color"] for c in classes], labels=[c["name"] for c in classes])
            relative = f"visualizations/{row['sample_id']}.jpg"
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            sheet([("RGB", rgb), ("UMD dominant GT", gt_im),
                   ("Prediction" if not failure else "Failure: empty prediction", pred_im)],
                  target, columns=3, width=640, height=520)
            cards.append(f"<section><p>{html.escape(row['sample_id'])}</p><a href='{relative}'><img loading='lazy' src='{relative}'></a></section>")
        print(f"[evaluate] {index}/{len(rows)} {row['sample_id']}", flush=True)
    timing_summary = {}
    for key in ["partglee_two_forward_seconds", "valen_predict_seconds"]:
        values = [r[key] for r in durations]
        timing_summary[key] = {
            "count": len(values), "mean": float(np.mean(values)) if values else None,
            "p50": float(np.percentile(values, 50)) if values else None,
            "p95": float(np.percentile(values, 95)) if values else None,
        }
    summary = {
        "dataset": "UMD", "metrics": total.summary(),
        "per_object": {name: value.summary() for name, value in sorted(per_object.items())},
        "failures": dict(failures), "manifest_sha256": sha(manifest_path),
        "inference_run": run, "observed_module_times": timing_summary,
        "protocol": {
            "rgb_only": True, "ground_truth_used_for_model_input": False,
            "whole_object_prohibition": False,
            "annotation": "manual gt_type; dominant gt_label; rank annotations unused",
            "ignore_gt_ids": [-1], "prediction_ids": "0 background/unmatched; 1..7 original UMD IDs",
            "foreground_mIoU": "Macro IoU over foreground classes with nonempty global union; absent-class false positives count",
            "gIoU_present_queries": "Mean IoU over GT-present (image, class) pairs; absent-class FP also captured by foreground_mIoU",
            "P_at_50_present_queries": "Fraction of GT-present pairs with IoU >= 0.5",
            "failures": "Recorded failures become all-background predictions; missing outputs require explicit allow_missing",
            "native_paper_metric_reproduction": False,
            "metric_note": "IoU/Dice implementation; original ranked-label / weighted-F protocols are not implemented here",
            "timing_note": "Observed calls include first-call effects; staged batch is not warmed end-to-end real-time latency",
        },
    }
    write(output / "metrics.json", summary)
    write_jsonl(output / "per_image.jsonl", per_image)
    write_jsonl(output / "per_query.jsonl", queries)
    with (output / "per_class.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["class", "class_id", "gt_pixels", "pred_pixels", "intersection", "union", "iou", "dice"])
        writer.writeheader()
        for name, row in summary["metrics"]["per_class"].items():
            writer.writerow({"class": name, **row})
    headline = summary["metrics"]
    metric_text = "\n".join(f"{key}: {headline[key]:.2%}" if headline[key] is not None else f"{key}: N/A"
                            for key in ["foreground_mIoU", "gIoU_present_queries", "P_at_50_present_queries", "mean_Dice_present_queries"])
    (output / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>UMD evaluation</title>"
        "<style>body{font-family:system-ui;margin:24px;background:white}img{max-width:100%}section{margin:20px 0}pre{white-space:pre-wrap}</style>"
        f"<h1>UMD 评测</h1><pre>{html.escape(metric_text)}</pre>"
        "<p>每图从左到右：RGB → 原始主要功能标签 → 预测。完整协议与失败计数见 <a href='metrics.json'>metrics.json</a>。</p>"
        + "\n".join(cards), encoding="utf-8")
    print(metric_text)
    print(f"Evaluation saved: {output / 'index.html'}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Compare saved UMD predictions with untouched original manual GT")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vis-limit", type=int, default=50)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    evaluate(args.manifest, args.predictions, args.output, args.vis_limit, args.allow_missing)


if __name__ == "__main__":
    main()
