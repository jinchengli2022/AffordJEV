"""Official Valen API, loaded once for a batch; RGB/definitions only."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import time
import traceback

from ..common import read, read_jsonl, write, sha


class Valen:
    def __init__(self, model_path, config):
        import torch
        from transformers import AutoModel
        self.torch = torch
        self.config = config
        self.model_path = str(Path(model_path).resolve(strict=True))
        if config["execution"] != "shared_state":
            raise ValueError("Use shared_state for the multi-region request")
        if not torch.cuda.is_available():
            raise RuntimeError("Valen requires CUDA in this baseline")
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
        started = time.perf_counter()
        self.model = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation=config["attn_implementation"],
        ).to("cuda").eval()
        if any(p.dtype != torch.float32 for p in self.model.head.parameters()):
            raise RuntimeError("Keep the official decision head in FP32")
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - started
        self.audit_out = None
        self.input_save_seconds = 0.0
        self.calls = 0
        if config.get("save_compiled_inputs", True):
            original_make_compiler = self.model.make_compiler

            def audited_compiler(*args, **kwargs):
                compiler = original_make_compiler(*args, **kwargs)
                original_compile = compiler.compile

                def compile_and_save(*compile_args, **compile_kwargs):
                    state = original_compile(*compile_args, **compile_kwargs)
                    saved_at = time.perf_counter()
                    folder = self.audit_out / "valen_compiled"
                    folder.mkdir(exist_ok=True)
                    if state.inputs is None:
                        raise ValueError("Expected shared-state compiler inputs")
                    tensors = {k: v.detach().cpu() for k, v in state.inputs.items() if isinstance(v, torch.Tensor)}
                    if self.config.get("save_compiled_tensors", True):
                        torch.save(tensors, folder / "inputs.pt")
                    prompt = compiler.tokenizer.decode(tensors["input_ids"][0].tolist(), skip_special_tokens=False)
                    (folder / "prompt.txt").write_text(prompt, encoding="utf-8")
                    write(folder / "metadata.json", {
                        "logical_tokens": state.logical_tokens, "compute_tokens": state.compute_tokens, "media": state.media,
                        "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in tensors.items()},
                        "questions": [{"id": q.qid, "keys": q.keys,
                                       "decision_position": q.branches[0].decision_position,
                                       "candidate_positions": q.branches[0].candidate_positions} for q in state.questions],
                    })
                    self.input_save_seconds += time.perf_counter() - saved_at
                    return state
                compiler.compile = compile_and_save
                return compiler
            self.model.make_compiler = audited_compiler

    def predict(self, out):
        torch = self.torch
        out = Path(out)
        for name, digest in read(out / "prepared_assets.json").items():
            if sha(out / name) != digest:
                raise ValueError(f"Prepared asset changed: {name}")
        request = read(out / "valen_request.json")
        self.audit_out = out
        self.input_save_seconds = 0.0
        if not request["questions"]:
            response = {"answers": {}, "skipped": "no_candidates", "usage": {"output_tokens": 0}}
            predict_seconds = 0.0
            peak = reserved = 0.0
        else:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.inference_mode():
                response = self.model.predict(request, media_root=str(out), execution="shared_state",
                                              temperature=self.config["temperature"])
            torch.cuda.synchronize()
            predict_seconds = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated() / 1024**2
            reserved = torch.cuda.max_memory_reserved() / 1024**2
        response["request_sha256"] = sha(out / "valen_request.json")
        write(out / "valen_response.json", response)
        write(out / "valen_runtime.json", {
            "model_path": self.model_path, "model_config_sha256": sha(Path(self.model_path) / "config.json"),
            "execution": "shared_state", "gpu": torch.cuda.get_device_name(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "model_reused_across_batch": True, "call_index": self.calls,
            "first_model_request": self.calls == 0,
            "backbone_dtype": str(next(self.model.backbone.parameters()).dtype),
            "head_dtype": str(next(self.model.head.parameters()).dtype),
            "packages": {k: importlib.metadata.version(k) for k in ["torch", "transformers", "Pillow"]},
            "timings": {"predict_seconds": predict_seconds,
                        "compiled_input_save_seconds_inside_predict": self.input_save_seconds},
            "timing_note": "Includes preprocessing, shared-state forward and input dump; excludes model load. No explicit warmup.",
            "peak_allocated_mib": peak, "peak_reserved_mib": reserved,
            "request_questions": len(request["questions"]),
            "output_tokens": response.get("usage", {}).get("output_tokens"),
        })
        if request["questions"]:
            self.calls += 1


def main():
    parser = argparse.ArgumentParser(description="Reusable Valen decision worker")
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--adapter", help="Optional local LoRA + decision-head checkpoint")
    args = parser.parse_args()
    jobs_path = Path(args.jobs).resolve()
    jobs = read_jsonl(jobs_path)
    jobs = [job for job in jobs if read(Path(job["output"]) / "run.json")["status"] == "prepared"]
    if not jobs:
        print("PROGRESS Valen 0/0 no prepared samples", flush=True)
        return
    config = read(Path(jobs[0]["output"]) / "config_snapshot.json")["valen"]
    model = Valen(jobs[0]["valen_model"], config)
    if args.adapter:
        from ..experiments.umd_valen.infer import load_adapter
        load_adapter(model.model, args.adapter)
        checkpoint = Path(args.adapter).resolve(strict=True)
        write(jobs_path.parent / "checkpoint_used.json", {
            "checkpoint": str(checkpoint),
            "head_sha256": sha(checkpoint / "head.safetensors"),
            "adapter_sha256": sha(checkpoint / "adapter/adapter_model.safetensors"),
        })
    write(jobs_path.parent / "valen_load.json", {
        "model_load_seconds_once": model.load_seconds, "model_instances": 1, "images": len(jobs),
    })
    for index, job in enumerate(jobs, 1):
        out = Path(job["output"])
        try:
            if read(out / "config_snapshot.json")["valen"] != config:
                raise ValueError("Batch decision config changed")
            model.predict(out)
            state = "classified"
            write(out / "run.json", {"status": state, "reference_annotations_used": False})
        except Exception as exc:
            traceback.print_exc()
            state = "failed"
            write(out / "run.json", {"status": state, "stage": "decision", "error": str(exc),
                  "reference_annotations_used": False})
        print(f"PROGRESS Valen {index}/{len(jobs)} {job['sample_id']} {state}", flush=True)


if __name__ == "__main__":
    main()
