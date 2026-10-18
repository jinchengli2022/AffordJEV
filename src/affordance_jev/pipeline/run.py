"""Offline UMD baseline: two reusable model workers, no GT in model jobs."""
import argparse
from collections import Counter
import html
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
import traceback

from PIL import Image

from ..common import read, write, read_jsonl, write_jsonl, sha, validate_classes
from .artifacts import merge_results


def sample_directory(root, sample_id):
    path = PurePosixPath(sample_id)
    if not sample_id or path.is_absolute() or ".." in path.parts or "\\" in sample_id or str(path) != sample_id:
        raise ValueError(f"Invalid sample_id: {sample_id!r}")
    return Path(root) / "samples" / Path(*path.parts)


def run_worker(python, module, jobs_path, log_path, extra_args=()):
    environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
    source = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
    with Path(log_path).open("w") as log:
        process = subprocess.Popen(
            [str(python), "-m", module, "--jobs", str(jobs_path), *map(str, extra_args)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=environment,
        )
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                if line.startswith("PROGRESS"):
                    print(line.strip(), flush=True)
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if code:
        raise RuntimeError(f"{module} exited with {code}; see {log_path}")


def run(args):
    started = time.perf_counter()
    manifest = Path(args.manifest).resolve(strict=True)
    rows = read_jsonl(manifest)
    if not rows or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Manifest must contain unique, nonempty samples")
    meta_path = manifest.with_suffix(".meta.json")
    meta = read(meta_path)
    if meta["manifest_sha256"] != sha(manifest):
        raise ValueError("Manifest does not match its indexing metadata")
    rgb_only = meta.get("subset") == "rgb"
    if meta["subset"] not in ("tools", "rgb"):
        raise ValueError("This first baseline assumes one tool per RGB; clutter needs multi-object inference")
    if not rgb_only and any(row.get("gt_type") != "manual" for row in rows):
        raise ValueError("UMD baseline requires verified manual-frame indexing")
    if args.limit < 0:
        raise ValueError("limit must be nonnegative")
    if args.limit:
        rows = rows[:args.limit]
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output must be empty: {out}")
    config, definitions = read(args.config), read(args.definitions)
    classes = validate_classes(definitions)
    expected = {0: "none", 1: "grasp", 2: "cut", 3: "scoop", 4: "contain",
                5: "pound", 6: "support", 7: "wrap-grasp"}
    if not rgb_only and {c["id"]: c["name"] for c in classes} != expected:
        raise ValueError("Use UMD native class IDs")
    if not rgb_only and config["partition"].get("whole_object_fraction") is not None:
        raise ValueError("Native UMD baseline disables the whole-object prohibition")
    for name in ["partglee_root", "partglee_weights", "valen_model", "partglee_python", "valen_python"]:
        setattr(args, name, str(Path(getattr(args, name)).resolve(strict=True)))
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    write(out / "config_snapshot.json", config)
    write(out / "affordances_snapshot.json", definitions)
    write(out / "manifest_metadata.json", meta)
    write_jsonl(out / "evaluation_manifest.jsonl", rows)
    write(out / "source_sha256.json", {
        str(p.relative_to(Path(__file__).resolve().parents[1])): sha(p)
        for p in sorted(Path(__file__).resolve().parents[1].rglob("*.py"))
    })
    jobs = []
    for row in rows:
        sample = sample_directory(out, row["sample_id"])
        sample.mkdir(parents=True)
        image_path = Path(row["rgb"]).resolve(strict=True)
        if sha(image_path) != row["rgb_sha256"]:
            raise ValueError(f"RGB changed after indexing: {row['sample_id']}")
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            if list(rgb.size) != row["size_wh"]:
                raise ValueError("Indexed RGB dimensions changed")
            rgb.save(sample / "rgb.png")
        write(sample / "config_snapshot.json", config)
        write(sample / "affordances_snapshot.json", definitions)
        # Deliberately exclude annotation paths, true class lists, object names as prompts, and depth.
        job = {
            "sample_id": row["sample_id"], "source_rgb": str(image_path), "output": str(sample),
            "source_sha256": row["rgb_sha256"], "rgb_sha256": sha(sample / "rgb.png"),
            "source_size_wh": row["size_wh"], "reference_annotations_used": False,
            "partglee_root": args.partglee_root, "partglee_weights": args.partglee_weights,
            "valen_model": args.valen_model,
        }
        write(sample / "job.json", job)
        write(sample / "run.json", {"status": "pending", "reference_annotations_used": False})
        jobs.append(job)
    jobs_path = out / "model_jobs.jsonl"
    write_jsonl(jobs_path, jobs)
    write(out / "run.json", {"status": "running", "samples": len(jobs),
          "manifest_sha256": sha(manifest), "evaluation_manifest_sha256": sha(out / "evaluation_manifest.jsonl"),
          "reference_annotations_used_in_inference": False})
    error = None
    stage_times = {}
    try:
        for stage, python, module in [
            ("segmentation", args.partglee_python, "affordance_jev.segmentation.partglee"),
            ("decision", args.valen_python, "affordance_jev.decision.valen"),
        ]:
            stage_start = time.perf_counter()
            print(f"[{stage}] {len(jobs)} RGB; one model load for this batch", flush=True)
            extra = ["--adapter", args.adapter] if stage == "decision" and getattr(args, "adapter", None) else []
            run_worker(python, module, jobs_path, out / f"logs/{stage}.log", extra)
            stage_times[stage] = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        for index, job in enumerate(jobs, 1):
            sample = Path(job["output"])
            if read(sample / "run.json")["status"] != "classified":
                continue
            try:
                merge_results(sample)
            except Exception as exc:
                traceback.print_exc()
                write(sample / "run.json", {"status": "failed", "stage": "merge", "error": str(exc),
                      "reference_annotations_used": False})
            print(f"[merge] {index}/{len(jobs)} {job['sample_id']}", flush=True)
        stage_times["merge"] = time.perf_counter() - stage_start
    except Exception as exc:
        error = str(exc)
        traceback.print_exc()
    counts = Counter(read(Path(job["output"]) / "run.json")["status"] for job in jobs)
    success = counts.get("complete", 0) == len(jobs)
    summary = {
        "status": "complete" if success else "failed", "samples": len(jobs), "counts": dict(counts),
        "error": error, "manifest_sha256": sha(manifest),
        "evaluation_manifest_sha256": sha(out / "evaluation_manifest.jsonl"),
        "whole_object_prohibition": False, "reference_annotations_used_in_inference": False,
        "limited_run": bool(args.limit or meta.get("limit") or meta.get("allow_incomplete")),
        "batch_wall_seconds": time.perf_counter() - started, "stage_wall_seconds": stage_times,
        "timing_scope": "Offline staged batch, including imports, model loading, inference and intermediate saving; not live latency.",
    }
    write(out / "run.json", summary)
    cards = []
    for job in jobs:
        sample = Path(job["output"])
        state = read(sample / "run.json")["status"]
        relative = sample.relative_to(out).as_posix()
        preview = f"<img loading='lazy' src='{relative}/result/affordance_overlay.png'>" if state == "complete" else ""
        cards.append(f"<section><a href='{relative}/index.html'>{html.escape(job['sample_id'])}</a> — {state}{preview}</section>")
    (out / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>UMD / PartGLEE + Valen</title>"
        "<style>body{font-family:system-ui;margin:24px;background:white}section{margin:18px 0}img{display:block;max-width:640px;width:100%}</style>"
        "<h1>UMD 推理预览</h1><p>RGB → PartGLEE 候选 → Valen 判断；不使用 GT。</p>"
        + "\n".join(cards), encoding="utf-8")
    print(f"Finished: {dict(counts)}; preview: {out / 'index.html'}")
    return 0 if success else 1


def main():
    parser = argparse.ArgumentParser(description="UMD RGB-only PartGLEE + Valen baseline")
    parser.add_argument("--adapter", help="Optional local LoRA + decision-head checkpoint; use SFT definitions")
    for name in ["manifest", "output", "config", "definitions", "partglee-root",
                 "partglee-weights", "valen-model", "partglee-python", "valen-python"]:
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--limit", type=int, default=0, help="0=all selected samples; positive values are development subsets")
    sys.exit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
