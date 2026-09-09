# GitHub release guide

Project repository: [Hisernberg/mri-conference-research](https://github.com/Hisernberg/mri-conference-research), with private visibility for the initial research release. The versioned [v1.0.0 release](https://github.com/Hisernberg/mri-conference-research/releases/tag/v1.0.0) provides the complete archive, both research notebooks, the seven report PDFs and SHA-256 checksums. Repository access requires the owner's GitHub account or an explicit collaborator invitation.

Start with the root README, [notebook index](../notebooks/README.md), [Kaggle run index](../kaggle/README.md) and [results index](../results/README.md). Kaggle runs remain private under `dasshovon`; their access is separate from GitHub repository access. The publication preserves the reported experiments and frozen source snapshots.

## Repository contents

| Path | Purpose |
|---|---|
| `README.md` | Entry point and honest current result status |
| `EXPERIMENT_PLAN.md` | Protocol and dated amendments |
| `docs/` | Dataset card, notebook audit, operation guide, results and PDF copies |
| `notebooks/` | Two generated research notebooks |
| `classification/` | Readable classification models and grouped experiment runner |
| `segmentation/src/qmmf/` | Reviewed segmentation library |
| `segmentation/scripts/conference_run.py` | CPU preparation and private GPU development workflow |
| `tests/`, `segmentation/tests/` | Scientific invariants and synthetic integration tests |
| `scripts/` | Notebook generation, authorized Kaggle API operations and document/release tooling |
| `results/` | Curated aggregate results, manifests and verification evidence |
| `audit/` | Original hashes, extracted code/output evidence and audit provenance |
| `kaggle/` | Generated submission bundles and private kernel metadata |

Full MRI volumes, classification images, caches, credentials, large checkpoints and complete downloaded Kaggle runtime folders are excluded by `.gitignore`. Checkpoints should be distributed separately only if appropriate, with a model card and checksum; they should not be force-added to Git history. The verified derived MSD single-slice panels carry a separate CC BY-SA 4.0 attribution notice in `results/segmentation_protected/qualitative/ATTRIBUTION.md`. Preserve that notice with the figures; it does not establish a license for the research code or classification images.

## Reproducibility record

For each reported experiment, retain the Kaggle URL and version, source hash, complete resolved configuration, data fingerprint, split hash, package/hardware environment, seed, selected checkpoint epoch, completed-run count, OOF or case-level prediction/metric files, and uncertainty calculation. Distinguish historical notebook results from new Kaggle execution and synthetic tests.

`audit/source_snapshots/` preserves exact submitted source dictionaries and execution cells. This lets later source improvements coexist with an auditable record of the code behind earlier results. Retain frozen group maps, exposure exclusions, per-case/per-group scores and cache conversion inventories. The release builder includes curated per-run JSON/CSV records while excluding binary checkpoints and MRI arrays.

Preserve the retrospective original-notebook exposure audit and `audit/historical_exposure_sensitivity_protocol.json`. Seven reserved groups include cases from earlier recorded validation. The supplementary 15-group analysis must remain labeled exploratory, with its freeze timing and the complete 51-group results retained. The historical audit and sensitivity analysis read source/metadata and verified metric tables; they add no MRI inference or training.

This release verifies 105 classification fits, three corrected segmentation pilot fits, 27 main segmentation fits and 12 reserved checkpoint evaluations. Main segmentation uses three seeds on grouped development fold 0; it is not four-fold cross-validation. The complete reserved report includes verified qualitative panels and the frozen historical sensitivity. Retain the negative quality-conditioning result and empty-reference failures. Do not publish an accuracy target, invented result, unsupported novelty claim or an inferred conference acceptance claim.

The dataset documentation records current metadata licenses. Source-code authorship and license are not specified by the supplied files; choose a code license only after the authors confirm the rights to the embedded code. Do not automatically apply a permissive code license to the MRI datasets or third-party implementations.

## Clone and reproduce the release

With Git and the GitHub CLI installed, authenticate using your own account and clone the private repository:

```bash
gh auth login
gh repo clone Hisernberg/mri-conference-research
cd mri-conference-research
git checkout v1.0.0
```

Install `requirements-dev.txt` in a suitable local Python environment, then run the verification suite, regenerate the notebooks from the reviewed source, regenerate the document PDFs, and inspect the release-file inventory. PDF conversion also requires the LibreOffice command-line application. Keep original notebook provenance, but use cleared generated notebooks for source publication; separately release executed artifacts that were inspected for credentials and private paths. Rebuilding after source changes can produce a new source hash; preserve the submitted snapshots that support the reported results.

```bash
PYTHONPATH=segmentation/src python -m pytest tests segmentation/tests -q
python scripts/build_notebooks.py classification
python scripts/build_notebooks.py preparation
python scripts/build_notebooks.py segmentation \
  --scope study --seed 42 --fold 0 \
  --epochs 120 --steps 100 --val-every 10 \
  --budget-seconds 29000 --slug mri-segmentation-study-s42
python scripts/render_documents.py
libreoffice -env:UserInstallation=file:///tmp/mri-docs-libreoffice \
  --headless --convert-to pdf:writer_pdf_Export \
  --outdir docs/pdf docs/docx/*.docx
python scripts/build_release.py
```

Authenticate Kaggle with an environment variable or external token file. Never paste a token into an `.ipynb`, Markdown document, kernel metadata file, commit message or screenshot. The provided token is stored outside this repository in the current local workspace.

## Prepare later updates

Preserve the released tag and its exact experiment snapshots. Prepare later changes on a branch, inspect the staged contents and push that branch to the same private repository. Publishing a later result requires its own completed evidence and updated release notes.

```bash
git switch main
git switch -c docs/update-research-record
git add README.md EXPERIMENT_PLAN.md .gitignore .gitattributes \
  requirements.txt requirements-dev.txt
git add docs notebooks classification segmentation scripts tests results audit kaggle
git diff --cached --stat
git diff --cached --check
git commit -m "Update MRI research documentation and evidence"
git push -u origin docs/update-research-record
```

Before running the broad `git add` commands, ensure the generated release inventory and secret check have passed. No credential file, raw dataset, derived volume cache or large model file belongs in the staging area.

The release builder writes `../mri_conference_release.zip` with an embedded file inventory and scans for Kaggle and GitHub credential patterns. Keep the downloadable archive as a release asset rather than duplicating it in Git history. GitHub releases also provide source archives for the tagged commit.

## Suggested repository description and paper positioning

Suggested description: “Audited MRI classification and missing-modality segmentation experiments with grouped/case splits, reproducible Kaggle execution and explicit ablation evidence.”

The main research direction is reliable incomplete-modality segmentation. The completed three-seed ablations and reserved evaluation do not support a quality-conditioning benefit; the no-quality control performs better on the reserved endpoints. The small image-classification benchmark is separate supporting work. A broader conference claim still needs strong current comparator implementations, independent cohort validation or a carefully limited claim, and an author-led assessment of novelty. Do not combine the two datasets into one headline performance number.

Add author names, affiliations, contact information, chosen code license and paper citation when those details exist. Avoid invented authors or a placeholder DOI presented as a real citation. A release tag should identify a frozen protocol and actual completed artifacts, with subsequent amendments recorded separately.
