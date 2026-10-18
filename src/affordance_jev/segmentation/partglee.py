"""Runs in the existing partglee Conda environment; saves all model stages."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from ..common import read, write, load_mask, save_mask, sha, bbox, padded_bbox
from ..visualization import overlay, sheet
from .foreground import select_object
from ..paths import CLIP_MODEL


class PartGLEE:
    def __init__(self, root, checkpoint, vocabulary, out, clip_model=CLIP_MODEL):
        root, checkpoint, out = Path(root).resolve(), Path(checkpoint).resolve(), Path(out).resolve()
        clip_model = Path(clip_model).resolve(strict=True)
        # Always import this checkout's source, not another installed Detectron2 fork.
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(root / "projects/PartGLEE/partglee/models/pixel_decoder/ops"))
        import detectron2
        if not Path(detectron2.__file__).resolve().is_relative_to(root):
            raise RuntimeError("Another Detectron2 was imported before the bundled PartGLEE source")
        from detectron2.config import get_cfg
        from detectron2.modeling import build_model
        from detectron2.data.transforms import ResizeShortestEdge
        import detectron2.projects
        project = root / "projects/PartGLEE/partglee"
        name = "detectron2.projects.partglee"
        spec = importlib.util.spec_from_file_location(name, project/"__init__.py", submodule_search_locations=[str(project)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        cfg = get_cfg()
        module.add_partglee_config(cfg)
        cfg.merge_from_file(str(root/"projects/PartGLEE/configs/Inference/Swin-L/paco.yaml"))
        cfg.MODEL.DEVICE = "cuda"
        cfg.MODEL.TEXT.LOCAL_MODEL_PATH = str(clip_model)
        self.model = build_model(cfg).eval()
        state = torch.load(checkpoint, map_location="cpu")
        state = state.get("model", state)
        loaded = self.model.load_state_dict(state, strict=False)
        missing = sorted(set(loaded.missing_keys) & set(dict(self.model.named_parameters())))
        import MultiScaleDeformableAttention
        write(out/"checkpoint_loading.json", {"checkpoint": str(checkpoint), "source_root": str(root),
              "deformable_attention_extension": MultiScaleDeformableAttention.__file__,
              "detectron2_source": detectron2.__file__, "clip_model": str(clip_model), "missing_parameters": missing,
              "missing_keys": loaded.missing_keys, "unexpected_keys": loaded.unexpected_keys})
        if missing:
            raise RuntimeError(f"PartGLEE checkpoint has missing parameters: {missing[:10]}")
        core = self.model.partglee
        core.test_topk_per_image = {**core.test_topk_per_image, "custom": 100}
        core.test_object_inst_num = {**core.test_object_inst_num, "custom": 50}
        core.test_part_inst_num = {**core.test_part_inst_num, "custom": 50}
        self.vocabulary = vocabulary
        self.labels = vocabulary["objects"] + vocabulary["parts"]
        self.resize = ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST]*2, cfg.INPUT.MAX_SIZE_TEST)

    @torch.inference_mode()
    def predict(self, rgb, kind):
        torch.cuda.synchronize()
        started = time.perf_counter()
        rgb = np.asarray(rgb)
        h, w = rgb.shape[:2]
        resized = self.resize.get_transform(rgb).apply_image(rgb)
        tensor = torch.as_tensor(resized.astype("float32").transpose(2, 0, 1))
        images = self.model.preprocess_image([{"image": tensor, "height": h, "width": w}], "custom")
        nobj = len(self.vocabulary["objects"])
        output = self.model.partglee(images, {}, "custom", batch_name_list=self.labels, is_train=False,
            custom_object_categories_idx=list(range(nobj)), custom_part_categories_idx=list(range(nobj, len(self.labels))))
        query_slice, category_slice = (slice(0, 50), slice(0, nobj)) if kind == "objects" else (slice(50, 100), slice(nobj, None))
        logits = output["pred_logits"][0, query_slice, category_slice]
        masks = F.interpolate(output["pred_masks"][:, query_slice], size=images.tensor.shape[-2:], mode="bilinear", align_corners=False)
        ih, iw = images.image_sizes[0]
        masks = F.interpolate(masks[:, :, :ih, :iw], size=(h, w), mode="bilinear", align_corners=False)[0]
        binary = masks > 0
        cls_score, indices = logits.sigmoid().max(-1)
        quality = cls_score * ((masks.sigmoid()*binary).flatten(1).sum(-1) / binary.flatten(1).sum(-1).clamp_min(1))
        result = (binary.cpu().numpy(), quality.cpu().tolist(), [self.vocabulary[kind][i] for i in indices.cpu().tolist()])
        torch.cuda.synchronize()
        return (*result, time.perf_counter()-started)


def save_raw(out, kind, rgb, masks, scores, labels, crop=None):
    folder = out / "raw" / kind
    folder.mkdir(parents=True, exist_ok=True)
    records, panels = [], []
    for i, (mask, score, label) in enumerate(zip(masks, scores, labels)):
        relative = f"raw/{kind}/{i:03}.png"
        save_mask(out/relative, mask)
        records.append({"raw_id": i, "mask": relative, "quality": score, "label": label, "bbox_xyxy": bbox(mask), "area": int(mask.sum())})
        im = overlay(rgb, [mask], labels=[str(i)])
        panels.append((f"raw {i} / q={score:.3f}", im.crop(crop) if crop else im))
    metadata = "objects.json" if kind.startswith("object") else "parts.json"
    write(out/"raw"/metadata, {"candidates": records, "quality_note": "class sigmoid score × mean foreground mask sigmoid; not measured IoU"})
    sheet(panels, out/"raw"/f"{kind}.jpg", columns=5)



def prepare_object_parts(adapter, job):
    out = Path(job["output"])
    config = read(out/"config_snapshot.json")
    total = time.perf_counter()
    times = {}
    rgb = Image.open(out/"rgb.png").convert("RGB")
    w, h = rgb.size
    fg_dir = out/"foreground"
    fg_dir.mkdir(exist_ok=True)
    if job.get("object_mask"):
        foreground = load_mask(job["object_mask"], (h, w))
        fg_meta = {"source": "provided_mask", "source_path": job["object_mask"], "source_sha256": sha(job["object_mask"])}
        times["object_forward_seconds"] = 0.0
    else:
        masks, scores, labels, elapsed = adapter.predict(rgb, "objects")
        times["object_forward_seconds"] = elapsed
        started = time.perf_counter()
        save_raw(out, "object_candidates", rgb, masks, scores, labels)
        (fg_dir/"object_candidates.jpg").write_bytes((out/"raw/object_candidates.jpg").read_bytes())
        times["object_candidates_save_seconds"] = time.perf_counter()-started
        index, selection_trace = select_object(masks, scores, config["foreground"])
        write(fg_dir/"selection_trace.json", {"selected_raw_id": index, "candidates": selection_trace,
              "rule": "quality × sqrt(image_area_fraction) × Gaussian_center_weight × border_penalty_if_truncated",
              "assumption": "One intended target, normally complete and near the image center; no per-image class or functional labels used"})
        foreground = masks[index]
        fg_meta = {"source": "partglee_central_complete_object", "selected_raw_id": index, "quality": scores[index], "label": labels[index]}
    started = time.perf_counter()
    crop = padded_bbox(foreground, config["foreground"]["crop_padding_fraction"])
    x0, y0, x1, y1 = crop
    masked = np.array(rgb)
    masked[~foreground] = config["foreground"]["background_rgb"]
    input_crop = Image.fromarray(masked).crop(crop)
    save_mask(fg_dir/"object_mask.png", foreground)
    save_mask(fg_dir/"crop_mask.png", foreground[y0:y1, x0:x1])
    overlay(rgb, [foreground], labels=["object"]).save(fg_dir/"object_overlay.png")
    Image.fromarray(masked).save(fg_dir/"masked_full.png")
    input_crop.save(fg_dir/"input_crop.png")
    rgb.crop(crop).save(fg_dir/"rgb_crop.png")
    write(fg_dir/"metadata.json", {**fg_meta, "crop_xyxy": crop, "source_size_wh": [w, h], "mask_sha256": sha(fg_dir/"object_mask.png")})
    times["foreground_prepare_save_seconds"] = time.perf_counter()-started
    masks_crop, scores, labels, elapsed = adapter.predict(input_crop, "parts")
    times["part_forward_seconds"] = elapsed
    started = time.perf_counter()
    masks = np.zeros((len(masks_crop), h, w), bool)
    masks[:, y0:y1, x0:x1] = masks_crop
    np.savez_compressed(out/"raw/proposals.npz", masks=masks, qualities=np.array(scores), masks_crop=masks_crop)
    save_raw(out, "part_candidates", rgb, masks, scores, labels, crop)
    times["part_candidates_save_seconds"] = time.perf_counter()-started
    times["prepare_objects_parts_and_save_seconds"] = time.perf_counter()-total
    write(out/"partglee_runtime.json", {"timings": times, "torch": torch.__version__,
          "gpu": torch.cuda.get_device_name(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
          "raw_object_queries": 50, "raw_part_queries": len(masks),
          "reference_annotations_used": False, "two_pass_inference": True,
          "model_reused_across_batch": True})


def main():
    from ..common import read_jsonl
    from ..pipeline.artifacts import prepare_regions
    import traceback
    parser = argparse.ArgumentParser(description="Reusable PartGLEE worker for a UMD batch")
    parser.add_argument("--jobs", required=True)
    args = parser.parse_args()
    jobs_path = Path(args.jobs).resolve()
    jobs = read_jsonl(jobs_path)
    if not jobs:
        raise ValueError("No jobs")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("PartGLEE requires CUDA")
    first = jobs[0]
    config = read(Path(first["output"])/"config_snapshot.json")
    start = time.perf_counter()
    adapter = PartGLEE(Path(first["partglee_root"]), Path(first["partglee_weights"]), config["vocabulary"], jobs_path.parent)
    torch.cuda.synchronize()
    write(jobs_path.parent/"partglee_load.json", {"model_load_seconds_once": time.perf_counter()-start,
          "model_instances": 1, "images": len(jobs), "weights": first["partglee_weights"]})
    for index, job in enumerate(jobs, 1):
        out = Path(job["output"])
        try:
            if read(out/"config_snapshot.json") != config:
                raise ValueError("Batch config changed")
            prepare_object_parts(adapter, job)
            prepare_regions(job)
            state = "prepared"
        except Exception as exc:
            traceback.print_exc()
            state = "failed"
            write(out/"run.json", {"status": state, "stage": "segmentation", "error": str(exc),
                  "reference_annotations_used": False})
        print(f"PROGRESS PartGLEE {index}/{len(jobs)} {job['sample_id']} {state}", flush=True)


if __name__ == "__main__":
    main()
