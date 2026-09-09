#!/usr/bin/env python3
"""Build and inspect a GitHub-ready archive without raw data or credentials."""
from pathlib import Path
import hashlib
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOP = ["README.md", "EXPERIMENT_PLAN.md", ".gitignore", ".gitattributes",
       "requirements.txt", "requirements-dev.txt"]
DIRS = ["docs", "notebooks", "classification", "segmentation/src", "segmentation/tests", "segmentation/scripts",
        "segmentation/configs", "tests", "scripts", "results", "audit", "kaggle", "paper"]
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".npz", ".npy", ".nii", ".pyc", ".key", ".pem"}
FORBIDDEN_PARTS = {"__pycache__", ".pytest_cache", "cache", "cache_v2", "checkpoints", "qmmf_runs"}


def files():
    paths = [ROOT / p for p in TOP]
    for d in DIRS:
        paths.extend(p for p in (ROOT / d).rglob("*") if p.is_file())
    paths += [ROOT / "segmentation/README.md", ROOT / "segmentation/requirements.txt"]
    return sorted({p for p in paths if p.exists() and p.suffix not in FORBIDDEN_SUFFIXES
                   and not any(part in FORBIDDEN_PARTS for part in p.relative_to(ROOT).parts)
                   and not p.name.endswith(".nii.gz")})


def build():
    paths = files(); rows = []
    # Never print secret matches. The scan reports only an affected path.
    token_pattern = re.compile(
        rb"(?:KGAT_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,})")
    for p in paths:
        payload = p.read_bytes()
        if token_pattern.search(payload):
            raise RuntimeError(f"Credential-like material found in {p.relative_to(ROOT)}")
        if p.stat().st_size > 20 * 1024**2:
            raise RuntimeError(f"Unexpected large release file: {p.relative_to(ROOT)}")
        rows.append({"path": p.relative_to(ROOT).as_posix(), "bytes": len(payload),
                     "sha256": hashlib.sha256(payload).hexdigest()})
    dest = ROOT.parent / "mri_conference_release.zip"
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, arcname="mri_conference/" + p.relative_to(ROOT).as_posix())
        z.writestr("mri_conference/RELEASE_INVENTORY.json", json.dumps(rows, indent=2))
    inventory = ROOT / "runtime/release_inventory.json"
    inventory.parent.mkdir(parents=True, exist_ok=True)
    inventory.write_text(json.dumps({"archive": str(dest), "files": rows,
        "archive_sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}, indent=2))
    print(json.dumps({"archive": str(dest), "files": len(paths), "bytes": dest.stat().st_size,
                      "credential_scan": "passed", "raw_data_and_checkpoints": "excluded"}, indent=2))


if __name__ == "__main__":
    build()
