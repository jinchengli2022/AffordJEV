"""Repository-local assets. Relative training paths are rooted at this checkout."""
import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
PARTGLEE_ROOT = PROJECT / "third_party/PartGLEE"
PARTGLEE_WEIGHTS = PROJECT / "models/partglee/PartGLEE_SwinL.pth"
CLIP_MODEL = PROJECT / "models/clip_vit_base_patch32"
VALEN_MODEL = PROJECT / "models/valen"


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT / path


def training_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("model_path", "records", "output", "checkpoint_dir", "warmup_head", "views"):
        if config.get(key):
            config[key] = str(project_path(config[key]).resolve())
    return config


def conda_python(name):
    conda = Path(os.environ.get("CONDA_EXE", Path.home() / "miniconda3/bin/conda"))
    return conda.parent.parent / "envs" / name / "bin/python"
