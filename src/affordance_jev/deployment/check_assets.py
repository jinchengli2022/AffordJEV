"""Verify exact local model assets before inference (no network/downloads)."""
import argparse
import hashlib
import json
from pathlib import Path

from ..paths import PROJECT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT)
    parser.add_argument("--sha256", action="store_true", help="Read all tensor bytes; size-only otherwise")
    parser.add_argument("--checkpoint", choices=["s0", "s0r", "a0", "a1", "b0", "b1", "c0", "c1"])
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "models/manifest.json").read_text())
    entries = list(manifest["base_assets"])
    if args.checkpoint:
        entries += manifest["optional_checkpoints"][args.checkpoint]
    failures = []
    for item in entries:
        path = root / item["path"]
        reason = None
        if not path.is_file():
            reason = "missing"
        elif path.stat().st_size != item["bytes"]:
            reason = "size mismatch"
        elif args.sha256:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != item["sha256"]:
                reason = "SHA256 mismatch"
        if reason:
            failures.append({"path": item["path"], "error": reason})
    print(json.dumps({"root": str(root), "checked": len(entries), "sha256": args.sha256,
                      "status": "failed" if failures else "passed", "failures": failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
