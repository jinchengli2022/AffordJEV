"""Read-only checks of the two external Conda environments; no weights loaded."""
import argparse
import importlib
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import platform
import sys

from ..paths import PROJECT, PARTGLEE_ROOT


def check(component):
    result = {"component": component, "python": platform.python_version(),
              "platform": f"{platform.system()} {platform.machine()}",
              "versions": {}, "warnings": [], "errors": []}
    expected_python = "3.10.22" if component == "partglee" else "3.11.16"
    if platform.python_version() != expected_python:
        result["errors"].append(f"Python {expected_python} required by the recorded environment")
    if (platform.system(), platform.machine()) != ("Linux", "x86_64"):
        result["errors"].append("Only Linux x86_64 is covered by these dependency pins")
    requirements = PROJECT / "requirements" / f"{component}.txt"
    pins = {}
    for line in requirements.read_text().splitlines():
        if line and not line.startswith(("#", "-")):
            name, expected = line.split("==", 1)
            pins[name] = expected
    pins.update({"torch": "2.2.2+cu121", "torchvision": "0.17.2+cu121"} if component == "partglee"
                else {"torch": "2.7.1+cu126", "torchvision": "0.22.1+cu126"})
    for name, expected in pins.items():
        try:
            actual = metadata.version(name)
            result["versions"][name] = actual
            if actual != expected:
                result["errors"].append(f"{name}: expected {expected}, found {actual}")
        except metadata.PackageNotFoundError:
            result["errors"].append(f"Missing package: {name}=={expected}")

    try:
        import torch
        import torchvision
        import cv2
        result["opencv_runtime"] = cv2.__version__
        if cv2.__version__ != "4.11.0":
            result["errors"].append("Loaded cv2 does not match the observed 4.11.0 runtime")
        result["cuda"] = {"torch_runtime": torch.version.cuda,
                          "available": torch.cuda.is_available()}
        if torch.cuda.is_available():
            result["cuda"]["device"] = torch.cuda.get_device_name()
            result["cuda"]["capability"] = list(torch.cuda.get_device_capability())
            if component == "valen" and not torch.cuda.is_bf16_supported():
                result["errors"].append("The selected GPU does not support the BF16 Valen path")
        else:
            result["errors"].append("CUDA is unavailable; check the driver and CUDA_VISIBLE_DEVICES")
    except Exception as exc:
        result["errors"].append(f"Runtime import failed: {type(exc).__name__}: {exc}")

    try:
        if component == "partglee":
            sys.path.insert(0, str(PARTGLEE_ROOT))
            sys.path.insert(0, str(PARTGLEE_ROOT / "projects/PartGLEE/partglee/models/pixel_decoder/ops"))
            import detectron2
            if not Path(detectron2.__file__).resolve().is_relative_to(PARTGLEE_ROOT.resolve()):
                raise RuntimeError("Detectron2 was not loaded from this checkout's PartGLEE source")
            from detectron2 import _C
            ops = importlib.import_module("MultiScaleDeformableAttention")
            result["extensions"] = {"detectron2": Path(_C.__file__).name,
                                     "MultiScaleDeformableAttention": Path(ops.__file__).name}
            project = PARTGLEE_ROOT / "projects/PartGLEE/partglee"
            name = "detectron2.projects.partglee"
            spec = importlib.util.spec_from_file_location(name, project / "__init__.py",
                                                         submodule_search_locations=[str(project)])
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            # Legacy Detectron2 metadata contains three unused tool dependencies.
            missing_tools = []
            for name in ("black", "future", "pydot"):
                try:
                    metadata.version(name)
                except metadata.PackageNotFoundError:
                    missing_tools.append(name)
            if missing_tools:
                result["warnings"].append("Unused legacy Detectron2 metadata dependencies absent: "
                                          + ", ".join(missing_tools)
                                          + "; plain pip check reports these as well")
        else:
            from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
            from peft import LoraConfig
            from transformers import AutoModel, AutoProcessor
            result["attention_path"] = {
                "dense_attention": "sdpa (selected by the pipeline)",
                "linear_attention": (qwen.chunk_gated_delta_rule.__module__
                                     if qwen.chunk_gated_delta_rule else "torch fallback"),
                "causal_conv1d": qwen.causal_conv1d_fn is not None,
                "complete_fast_path": bool(qwen.is_fast_path_available),
            }
            if qwen.chunk_gated_delta_rule is None or not qwen.chunk_gated_delta_rule.__module__.startswith("fla."):
                result["errors"].append("The recorded Valen path uses FLA gated-delta attention")
            if qwen.causal_conv1d_fn is not None:
                result["errors"].append("causal-conv1d changes the recorded torch convolution fallback")
            try:
                metadata.version("flash-attn")
                result["warnings"].append("flash-attn is installed; keep the pipeline's SDPA setting")
            except metadata.PackageNotFoundError:
                pass
    except Exception as exc:
        result["errors"].append(f"Model/extension import failed: {type(exc).__name__}: {exc}")
    result["ok"] = not result["errors"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("partglee", "valen"), required=True)
    args = parser.parse_args()
    result = check(args.component)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
