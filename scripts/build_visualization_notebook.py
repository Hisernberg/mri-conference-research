#!/usr/bin/env python3
"""Package immutable result inputs into a credential-free Kaggle CPU notebook."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import shutil
import subprocess
import zipfile
import nbformat as nbf
from build_conference_figures import required_inputs

ROOT = Path(__file__).resolve().parents[1]


def setup_cell(payload_sha, manifest_sha):
    header = f"BUNDLE_SHA256 = {payload_sha!r}\nMANIFEST_SHA256 = {manifest_sha!r}\n"
    return header + '''from pathlib import Path
import hashlib, io, json, shutil, zipfile
import nbformat as nbf
from nbclient import NotebookClient
from IPython.display import display, Image
WORKSPACE = (Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd() / 'runtime') / 'conference_figure_workspace'
WORKSPACE.mkdir(parents=True, exist_ok=True)
INPUT = Path('/kaggle/input') if Path('/kaggle/input').exists() else Path.cwd() / 'runtime/conference_evidence_dataset'
archives = list(INPUT.rglob('evidence_payload.zip'))
if archives:
    assert len(archives) == 1
    payload = archives[0].read_bytes()
    assert hashlib.sha256(payload).hexdigest() == BUNDLE_SHA256
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():
            path = (WORKSPACE / name).resolve()
            assert path.is_relative_to(WORKSPACE.resolve())
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(name))
else:
    # Kaggle may expand an uploaded ZIP. Validate the manifest and each file.
    manifests = list(INPUT.rglob('EVIDENCE_MANIFEST.json'))
    assert len(manifests) == 1, 'Attach the private mri-conference-figure-evidence dataset.'
    source = manifests[0].parent
    assert hashlib.sha256(manifests[0].read_bytes()).hexdigest() == MANIFEST_SHA256
    evidence = json.loads(manifests[0].read_text())
    for name in [*evidence['files'], 'EVIDENCE_MANIFEST.json']:
        path = (WORKSPACE / name).resolve()
        assert path.is_relative_to(WORKSPACE.resolve())
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, path)
assert hashlib.sha256((WORKSPACE / 'EVIDENCE_MANIFEST.json').read_bytes()).hexdigest() == MANIFEST_SHA256
evidence = json.loads((WORKSPACE / 'EVIDENCE_MANIFEST.json').read_text())
for name, digest in evidence['files'].items():
    assert hashlib.sha256((WORKSPACE / name).read_bytes()).hexdigest() == digest, name
print('Evidence bundle verified:', len(evidence['files']), 'files; SHA-256', BUNDLE_SHA256)
'''


def build(catalog):
    specification = json.loads(catalog.read_text())
    code = ROOT / "scripts/build_conference_figures.py"
    assert specification["generator_sha256"] == hashlib.sha256(code.read_bytes()).hexdigest()
    source_paths = required_inputs() + ["scripts/build_conference_figures.py"]
    manifest = {"evidence_base_git_commit": subprocess.check_output(["git", "rev-parse", "v1.0.0"], cwd=ROOT, text=True).strip(),
                "scope": "Frozen results and verified derived figures; no MRI arrays, model checkpoints or credentials.",
                "files": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_paths}}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_paths):
            item = zipfile.ZipInfo(path, date_time=(2026, 9, 9, 0, 0, 0)); item.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(item, (ROOT / path).read_bytes())
        item = zipfile.ZipInfo("EVIDENCE_MANIFEST.json", date_time=(2026, 9, 9, 0, 0, 0))
        item.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(item, json.dumps(manifest, indent=2))
    payload = stream.getvalue(); payload_sha = hashlib.sha256(payload).hexdigest()
    evidence_dir = ROOT / "runtime/conference_evidence_dataset"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "evidence_payload.zip").write_bytes(payload)
    dataset_metadata = {"id": "dasshovon/mri-conference-figure-evidence", "title": "MRI Conference Figure Evidence",
        "subtitle": "Frozen inputs for 20 audited MRI research visualizations",
        "description": "Private reproducibility bundle for the MRI conference visualization notebook. Contains frozen metric/audit tables, the plotting source and an attributed derived montage, not raw MRI volumes or checkpoints. The derived MSD montage retains CC BY-SA 4.0 terms in ATTRIBUTION.md. Source code has no new license grant; rights and terms remain those of the original sources. The Other license selection records these separate terms and does not apply an open license to the complete bundle.",
        "licenses": [{"name": "other"}]}
    (evidence_dir / "dataset-metadata.json").write_text(json.dumps(dataset_metadata, indent=2) + "\n")
    shutil.copy2(ROOT / "results/segmentation_protected/qualitative/ATTRIBUTION.md", evidence_dir / "ATTRIBUTION.md")
    report = nbf.v4.new_notebook(metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    report.cells = [nbf.v4.new_markdown_cell("# MRI conference figure report\n\nTwenty visualizations of frozen Kaggle evidence. This executed report performs no training or inference. Its source inputs and output files have SHA-256 inventories. The quality-conditioning hypothesis is not supported by the completed reserved comparisons.\n\nFigure 19 reuses the verified real-prediction montage with CC BY-SA 4.0 attribution; its PNG/PDF are unchanged. All other figures are generated from the recorded audit, prediction or metric tables."),
                    nbf.v4.new_code_cell("from pathlib import Path\nimport sys, json\nROOT = Path.cwd()\nsys.path.insert(0, str(ROOT / 'scripts'))\nfrom build_conference_figures import FigureBuilder\nOUT = ROOT / 'conference_figures'\nmanifest = FigureBuilder(ROOT, OUT).run()\nassert manifest['figure_count'] == 20\nfrom IPython.display import display, Image\n")]
    for record in specification["figures"]:
        report.cells += [nbf.v4.new_markdown_cell(f"## Figure {record['number']:02d}: {record['title']}\n\n{record['caption']}\n\n**Interpretation:** {record['interpretation']}\n\n**Writing placement:** {record['suggested_placement']}"),
                         nbf.v4.new_code_cell(f"display(Image(filename=str(OUT / {record['stem'] + '.png'!r})))")]
    report.cells += [nbf.v4.new_markdown_cell("## Attribution and scope\n\nThe derived MRI montage is covered by the bundled ATTRIBUTION.md notice. Group bootstrap intervals retain three fixed training seeds; seed SD is a different quantity. Groups are conservative image-similarity units, not verified patients. Simulated channel masking occurs after complete-modality preprocessing. Retain the historical exposure, exploratory sensitivity and ET-empty failures when using these figures.")]
    for i, cell in enumerate(report.cells): cell.id = f"figure-report-{i:02d}"
    outer = nbf.v4.new_notebook(metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"}})
    outer.cells = [nbf.v4.new_markdown_cell("# MRI conference: 20 verified visualizations\n\nRun this supporting **CPU notebook** to regenerate the conference figures from the frozen result tables. The two original research notebooks remain unchanged. Attach the private mri-conference-figure-evidence Kaggle dataset. This notebook verifies every input hash and needs no API credentials in its cells, GPU, training or inference. Locally, run scripts/build_visualization_notebook.py first to prepare the same input bundle.\n\nThe output contains 20 PNG/PDF/SVG figure sets, 20 plotted-data CSVs, source/output manifests and an actually executed gallery notebook. Scientific chart SVG/PDF files are vector exports; Figure 19 preserves the original raster montage and its attribution.\n\nThe companion gallery is executed with nbclient inside this Kaggle session, and every generated image is also displayed below. Read the full captions before transferring a figure into the paper."),
        nbf.v4.new_code_cell(setup_cell(payload_sha, hashlib.sha256(json.dumps(manifest, indent=2).encode()).hexdigest())),
        nbf.v4.new_code_cell("import time, shutil\nfrom datetime import datetime, timezone\n" +
            f"report = nbf.reads({nbf.writes(report)!r}, as_version=4)\n" +
            "started = time.monotonic()\nclient = NotebookClient(report, timeout=600, kernel_name='python3', resources={'metadata': {'path': str(WORKSPACE)}})\nexecuted = client.execute()\nnbf.validate(executed)\nassert all(cell.execution_count is not None for cell in executed.cells if cell.cell_type == 'code')\nassert not any(output.output_type == 'error' for cell in executed.cells if cell.cell_type == 'code' for output in cell.outputs)\nOUT = WORKSPACE / 'conference_figures'\nnbf.write(executed, OUT / '03_conference_visualizations.executed.ipynb')\nshutil.copy2(WORKSPACE / 'EVIDENCE_MANIFEST.json', OUT / 'EVIDENCE_MANIFEST.json')\nmanifest = json.loads((OUT / 'figure_manifest.json').read_text())\nassert manifest['figure_count'] == 20\nreceipt = {'completed': True, 'recorded_utc': datetime.now(timezone.utc).isoformat(), 'bundle_sha256': BUNDLE_SHA256, 'figure_count': 20, 'executed_code_cells': sum(c.cell_type == 'code' for c in executed.cells), 'wall_seconds': time.monotonic() - started, 'is_kaggle': Path('/kaggle/working').exists(), 'gpu_requested': False, 'training_performed': False, 'inference_performed': False, 'files': {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'execution_receipt.json'}}\n(OUT / 'execution_receipt.json').write_text(json.dumps(receipt, indent=2) + '\\n')\narchive = shutil.make_archive(str(WORKSPACE.parent / 'conference_visualizations'), 'zip', OUT)\nprint('Completed: 20 figures and executed notebook. Archive:', Path(archive).name)\nprint('Elapsed seconds:', round(receipt['wall_seconds'], 2))\nfor record in manifest['figures']:\n    print(f\"Figure {record['number']:02d}: {record['title']}\")\n    display(Image(filename=str(OUT / (record['stem'] + '.png'))))\n")]
    outer.cells[1].metadata["jupyter"] = {"source_hidden": True}
    for i, cell in enumerate(outer.cells): cell.id = f"conference-{i:02d}"
    nbf.validate(outer)
    target = ROOT / "kaggle/conference_visualizations"; target.mkdir(parents=True, exist_ok=True)
    path = target / "03_conference_visualizations.ipynb"; nbf.write(outer, path)
    shutil.copy2(path, ROOT / "notebooks/03_conference_visualizations.ipynb")
    metadata = {"id": "dasshovon/mri-conference-20-visualizations", "title": "MRI Conference 20 Visualizations",
                "code_file": path.name, "language": "python", "kernel_type": "notebook", "is_private": True,
                "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
                "dataset_sources": ["dasshovon/mri-conference-figure-evidence"], "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (target / "evidence-dataset-metadata.json").write_text(json.dumps(dataset_metadata, indent=2) + "\n")
    record = {"bundle_sha256": payload_sha, "bundle_bytes": len(payload), "notebook_bytes": path.stat().st_size,
              "notebook_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "evidence": manifest}
    (target / "bundle_record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "evidence"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=ROOT / "results/conference_figures/figure_manifest.json")
    args = parser.parse_args()
    build(args.catalog)
