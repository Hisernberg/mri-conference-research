#!/usr/bin/env python3
"""Build self-contained Kaggle notebooks from the readable source tree."""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def materializer(files):
    sources = {p.relative_to(ROOT).as_posix(): p.read_text() for p in sorted(files)}
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    code = """# This cell is generated from the readable Python source in the GitHub package.
import hashlib, json, os, sys
from pathlib import Path
WORK = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
SOURCE_ROOT = WORK / 'mri_source'
SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
"""
    code += "SOURCES = " + repr(sources) + "\n"
    code += f"SOURCE_SHA256 = {digest!r}\n"
    code += """assert hashlib.sha256(json.dumps(SOURCES, sort_keys=True).encode()).hexdigest() == SOURCE_SHA256
for rel, text in SOURCES.items():
    path = SOURCE_ROOT / rel
    assert path.resolve().is_relative_to(SOURCE_ROOT.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(SOURCE_ROOT / 'segmentation' / 'src'))
print('Source SHA256:', SOURCE_SHA256)
"""
    return code, digest


def build(kind, user="dasshovon", scope="pilot", run_options=None, kernel_slug=None):
    if kind == "io-benchmark":
        files = list((ROOT / "segmentation/src").rglob("*.py"))
        files += [ROOT / "segmentation/scripts/conference_run.py", ROOT / "segmentation/scripts/benchmark_io.py"]
        files += list((ROOT / "segmentation/configs/grouped_v2").glob("*.json"))
        code, digest = materializer(files)
        code += "\nimport subprocess\nsubprocess.run([sys.executable, str(SOURCE_ROOT / 'segmentation/scripts/benchmark_io.py')], check=True)\n"
        compile(code, "io_benchmark.py", "exec")
        dest = ROOT / "kaggle/io_benchmark"; dest.mkdir(parents=True, exist_ok=True)
        (dest / "io_benchmark.py").write_text(code)
        (dest / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/mri-segmentation-io-benchmark", "title": "MRI Segmentation IO Benchmark",
            "code_file": "io_benchmark.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": True, "enable_internet": False,
            "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
            "kernel_sources": [f"{user}/mri-segmentation-preparation-v2", f"{user}/mri-segmentation-training-cache"],
            "competition_sources": []}, indent=2))
        print("Training I/O benchmark source", digest)
        return
    if kind == "training-cache":
        files = list((ROOT / "segmentation/src").rglob("*.py"))
        files += [ROOT / "scripts/build_training_cache.py"]
        code, digest = materializer(files)
        code += "\nfrom scripts.build_training_cache import run\nrun()\n"
        compile(code, "training_cache.py", "exec")
        dest = ROOT / "kaggle/training_cache"; dest.mkdir(parents=True, exist_ok=True)
        (dest / "training_cache.py").write_text(code)
        (dest / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/mri-segmentation-training-cache", "title": "MRI Segmentation Training Cache",
            "code_file": "training_cache.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": False, "enable_internet": False,
            "machine_shape": "None", "dataset_sources": [],
            "kernel_sources": [f"{user}/mri-segmentation-preparation-v2"],
            "competition_sources": []}, indent=2))
        print("Lossless training-cache source", digest)
        return
    if kind == "preparation":
        files = list((ROOT / "segmentation" / "src").rglob("*.py"))
        files += [ROOT / "segmentation" / "scripts" / "conference_run.py"]
        code, digest = materializer(files)
        code += """
import importlib.util, subprocess
if importlib.util.find_spec('nibabel') is None:
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'nibabel==5.4.2'], check=True)
subprocess.run([sys.executable, str(SOURCE_ROOT / 'segmentation/scripts/conference_run.py'),
                '--input', '/kaggle/input', '--work', str(WORK), '--scope', 'prepare'], check=True)
"""
        compile(code, "prepare.py", "exec")
        dest = ROOT / "kaggle" / "preparation"; dest.mkdir(parents=True, exist_ok=True)
        (dest / "prepare.py").write_text(code)
        (dest / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/mri-segmentation-preparation-v2", "title": "MRI Segmentation Preparation v2",
            "code_file": "prepare.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": False, "enable_internet": True,
            "machine_shape": "None", "dataset_sources": ["thisisrick25/medical-segmentation-decathlon-brain-tumour"],
            "kernel_sources": [], "competition_sources": []}, indent=2))
        print("CPU preparation source", digest)
        return
    if kind == "classification":
        files = list((ROOT / "classification").glob("*.py"))
        slug, title, filename = "mri-classification-ablation-v2", "MRI Classification Ablation v2", "01_classification_ablation.ipynb"
        dataset = "navoneel/brain-mri-images-for-brain-tumor-detection"
        intro = """# MRI classification: grouped component ablations

Seven prespecified variants, five frozen grouped outer folds and three training seeds.
The data have no verified patient identifiers: these are **image-level exploratory results**.
Read `EXPERIMENT_PLAN.md`, `docs/NOTEBOOK_AUDIT.md` and the dataset documentation before reporting.
The source snapshot is included below so Kaggle runs exactly the reviewed Python code.
"""
        run = """from classification.study import main
roots = [p for p in Path('/kaggle/input').rglob('brain-mri-images-for-brain-tumor-detection') if p.is_dir()]
if len(roots) != 1:
    raise RuntimeError(f'Expected one attached classification dataset, found {roots}')
main(['--root', str(roots[0]), '--out', str(WORK / 'classification_results'),
      '--epochs', '60', '--patience', '12', '--batch', '32', '--seeds', '42', '43', '44'])
"""
    else:
        files = list((ROOT / "segmentation" / "src").rglob("*.py"))
        files += [ROOT / "segmentation" / "scripts" / "conference_run.py"]
        files += list((ROOT / "segmentation" / "tests").glob("*.py"))
        files += list((ROOT / "segmentation/configs/grouped_v2").glob("*.json"))
        slug, title, filename = "mri-segmentation-qmmf-v2", "MRI Segmentation QMMF v2", "02_segmentation_study.ipynb"
        dataset = "thisisrick25/medical-segmentation-decathlon-brain-tumour"
        intro = """# Corrected QMMF-Net segmentation study

Verified modality order, training-fitted inference normalization and true full-modality teacher input.
Frozen splits isolate conservative image-similarity groups; training and primary metrics weight groups equally.
Pilot outputs are development evidence. The locked test stays closed until the full protocol is frozen.
Read the experiment plan and audit for scope, baselines, ablations and limitations.
"""
        options = run_options or {}
        extra_args = []
        for option, value in options.items():
            if value is not None:
                extra_args += ["--" + option.replace("_", "-")]
                extra_args += [str(v) for v in value] if isinstance(value, list) else [str(value)]
        if "budget_seconds" not in options or options["budget_seconds"] is None:
            extra_args += ["--budget-seconds", "6000" if scope == "pilot" else "15600"]
        run = f"""import importlib.util, subprocess
if importlib.util.find_spec('nibabel') is None:
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'nibabel==5.4.2'], check=True)
subprocess.run([sys.executable, str(SOURCE_ROOT / 'segmentation/scripts/conference_run.py'),
                '--input', '/kaggle/input', '--work', str(WORK), '--scope', {scope!r}]
               + {extra_args!r}, check=True)
"""
    source, digest = materializer(files)
    if kernel_slug is not None:
        if kind != "segmentation" or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", kernel_slug):
            raise ValueError("A study kernel slug must contain lowercase letters, digits and internal hyphens")
        slug, title = kernel_slug, kernel_slug.replace("-", " ").title()
    nb = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(intro),
        nbf.v4.new_markdown_cell("## Exact source snapshot\nThe separate source files are the review/edit interface; this generated cell makes the notebook portable."),
        nbf.v4.new_code_cell(source), nbf.v4.new_markdown_cell("## Run the frozen protocol"),
        nbf.v4.new_code_cell(run)], metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "mri_source_sha256": digest})
    nbf.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == "code":
            compile(cell.source, filename, "exec")
    target = ROOT / "notebooks" / filename; nbf.write(nb, target)
    dest = ROOT / "kaggle" / kind; dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target, dest / filename)
    metadata = {"id": f"{user}/{slug}", "title": title, "code_file": filename,
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_internet": kind == "segmentation", "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [dataset] if kind == "classification" else [],
        "competition_sources": [],
        "kernel_sources": [] if kind == "classification" else [f"{user}/mri-segmentation-preparation-v2"], "model_sources": []}
    if kind == "segmentation" and scope == "study":
        metadata["kernel_sources"].append(f"{user}/mri-segmentation-training-cache")
    (dest / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    print(kind, digest, target)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("kind", choices=["classification", "segmentation", "preparation", "training-cache", "io-benchmark"])
    p.add_argument("--user", default="dasshovon"); p.add_argument("--scope", default="pilot")
    p.add_argument("--seed", type=int); p.add_argument("--fold", type=int)
    p.add_argument("--epochs", type=int); p.add_argument("--steps", type=int)
    p.add_argument("--val-every", type=int); p.add_argument("--budget-seconds", type=int)
    p.add_argument("--variants", nargs="+")
    p.add_argument("--slug", help="Separate private segmentation run identity, e.g. one immutable output per seed")
    a = p.parse_args()
    options = {key: getattr(a, key) for key in ["seed", "fold", "epochs", "steps", "val_every", "budget_seconds", "variants"]}
    build(a.kind, a.user, a.scope, options, a.slug)
