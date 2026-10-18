"""Run the same pipeline on RGB files without requiring dataset annotations."""
import argparse
from pathlib import Path
import sys
import tempfile

from PIL import Image

from ..common import sha, write, write_jsonl
from ..paths import PROJECT, PARTGLEE_ROOT, PARTGLEE_WEIGHTS, VALEN_MODEL, conda_python
from .run import run


def main():
    parser = argparse.ArgumentParser(description="RGB file/directory → PartGLEE → Valen → aligned masks")
    parser.add_argument("--input", required=True, help="One RGB image or a directory of images")
    parser.add_argument("--output", required=True, help="A new empty output directory")
    parser.add_argument("--config", default=str(PROJECT / "configs/umd_pipeline.json"))
    parser.add_argument("--definitions", default=str(PROJECT / "configs/umd_affordances.json"))
    parser.add_argument("--partglee-root", default=str(PARTGLEE_ROOT))
    parser.add_argument("--partglee-weights", default=str(PARTGLEE_WEIGHTS))
    parser.add_argument("--valen-model", default=str(VALEN_MODEL))
    parser.add_argument("--partglee-python", default=str(conda_python("partglee")))
    parser.add_argument("--valen-python", default=str(conda_python("valen-4b-sft")))
    parser.add_argument("--adapter", help="Optional local LoRA + head; use matching SFT definitions")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    source = Path(args.input).resolve(strict=True)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in extensions) if source.is_dir() else [source]
    if not images:
        raise ValueError(f"No RGB images in {source}")
    rows = []
    for path in images:
        with Image.open(path) as image:
            size = list(image.size)
        rows.append({"sample_id": path.relative_to(source).as_posix() if source.is_dir() else path.name,
                     "rgb": str(path), "rgb_sha256": sha(path), "size_wh": size})
    with tempfile.TemporaryDirectory(prefix="affordance-rgb-") as temporary:
        manifest = Path(temporary) / "rgb.jsonl"
        write_jsonl(manifest, rows)
        write(manifest.with_suffix(".meta.json"), {
            "dataset": "User RGB", "subset": "rgb", "samples": len(rows),
            "manifest_sha256": sha(manifest), "reference_annotations_used": False,
        })
        args.manifest = str(manifest)
        sys.exit(run(args))


if __name__ == "__main__":
    main()
