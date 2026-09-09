# MRI research: conference writing package

An audited study of quality conditioning for MRI segmentation with missing modalities, with a separate classification benchmark. The repository contains completed Kaggle experiments, **20 verified visualizations**, a manuscript starter, editable tables and the evidence needed to trace every reported result.

[Manuscript draft](docs/MANUSCRIPT_DRAFT.md) · [Writing guide](docs/WRITING_GUIDE.md) · [20-figure catalog](docs/FIGURE_CATALOG.md) · [Complete release v1.1.0](https://github.com/Hisernberg/mri-conference-research/releases/tag/v1.1.0)

**Main finding:** the completed reserved evaluation favors removing quality features. The package preserves negative results, inconclusive comparisons and failure cases. These internal experiments do not establish patient-level external validation, architectural novelty or conference acceptance.

## Start writing

| Need | Open this |
|---|---|
| A structured paper with actual results | [Manuscript Markdown](docs/MANUSCRIPT_DRAFT.md), [editable DOCX](docs/docx/MANUSCRIPT_DRAFT.docx), [review PDF](docs/pdf/MANUSCRIPT_DRAFT.pdf) |
| Figure captions and interpretation limits | [Figure catalog](docs/FIGURE_CATALOG.md), [catalog PDF](docs/pdf/FIGURE_CATALOG.pdf), [20-page figure atlas](results/conference_figures/conference_figure_atlas.pdf) |
| Figures as shown in the executed notebook | [Executed Kaggle gallery](results/conference_figures/03_conference_visualizations.executed.ipynb) |
| Tables for a conference template | [Eight table sets: CSV / Markdown / LaTeX](paper/tables/) |
| Statements tied to evidence | [Claim-evidence map](paper/claim_evidence.csv), [writing guide](docs/WRITING_GUIDE.md) |
| Citations and positioning | [BibTeX references](paper/references.bib), [related-work report](docs/RELATED_WORK.md) |
| Dataset and original-notebook analysis | [Dataset report](docs/DATASETS.md), [notebook audit](docs/NOTEBOOK_AUDIT.md), [experiment plan](EXPERIMENT_PLAN.md) |

No venue or page limit has been supplied. The draft is venue-neutral, with a suggested main-paper selection and all 20 figures available for the supplement. Authors still need to provide their identities, institutional declarations, source-code licensing decision and venue formatting.

## Completed experiments

| Study | Completed scope | Sampling units |
|---|---|---|
| Classification | 7 variants × 5 grouped folds × 3 seeds = **105 fits** | 228 unique images / 209 conservative groups |
| Corrected segmentation pilot | **3 fits** | Short-training feasibility evidence |
| Main segmentation | 9 variants × 3 seeds = **27 fits** | Development fold 0; 91 outer cases / 55 groups; 16 subset representatives |
| Reserved segmentation | 4 families × 3 checkpoints = **12 evaluations** | 66 cases / 51 groups; all 15 subsets on 51 representatives |
| Conference visualization | **20 figures**, 21 executed gallery code cells | Kaggle CPU, version 1; no new training or inference |

All original run URLs and exact versions are in the [Kaggle index](kaggle/README.md). The [new visualization notebook](https://www.kaggle.com/code/dasshovon/mri-conference-20-visualizations) completed on Kaggle CPU. Its input/output hashes, plotted CSV values and all 20 embedded image outputs were [independently verified](audit/conference_visualizations_verification.json).

## Reserved results

| Model | Full Dice | Mean-subset Dice | Worst-subset Dice |
| --- | --- | --- | --- |
| QMMF | 0.7689 +/- 0.0055 | 0.6322 +/- 0.0080 | 0.3672 +/- 0.0120 |
| No quality | 0.7769 +/- 0.0007 | 0.6405 +/- 0.0075 | 0.3886 +/- 0.0137 |
| HeMIS-style | 0.6837 +/- 0.0007 | 0.5462 +/- 0.0080 | 0.2748 +/- 0.0286 |
| U-Net 2.5D | 0.7621 +/- 0.0074 | 0.6034 +/- 0.0078 | 0.3140 +/- 0.0115 |

Values are mean +/- sample SD across the three fixed seeds. Full Dice uses 66 cases grouped into 51 units; subset metrics use 51 fixed representatives. QMMF minus no quality is **-0.0080**, adjusted paired group interval **[-0.0133, -0.0028]**, for full modalities. The mean- and worst-subset comparisons also favor no quality. See [all results](docs/RESULTS.md) and [the paired table](paper/tables/reserved_contrasts.md).

## Twenty figures from the Kaggle report

Each figure has PNG, PDF, SVG and a plotted-data CSV. Click its caption link for the complete interpretation and source files. Figures 01–18 and 20 are chart exports with vector PDF/SVG elements. Figure 19 preserves the original real-prediction montage; its SVG wraps the original raster. The [figure atlas](results/conference_figures/conference_figure_atlas.pdf) contains all 20 PDF pages.

### Suggested main-paper figures

**Figure 02. Frozen segmentation cohorts** — [caption and evidence](docs/FIGURE_CATALOG.md) · [PDF](results/conference_figures/fig02_segmentation_cohorts.pdf) · [CSV](results/conference_figures/data/fig02_segmentation_cohorts.csv)

![Figure 02: Frozen segmentation cohorts](results/conference_figures/fig02_segmentation_cohorts.png)

**Figure 08. Main paired ablation intervals** — [caption and evidence](docs/FIGURE_CATALOG.md) · [PDF](results/conference_figures/fig08_main_paired_ablations.pdf) · [CSV](results/conference_figures/data/fig08_main_paired_ablations.csv)

![Figure 08: Main paired ablation intervals](results/conference_figures/fig08_main_paired_ablations.png)

**Figure 11. Reserved evaluation: four selected model families** — [caption and evidence](docs/FIGURE_CATALOG.md) · [PDF](results/conference_figures/fig11_reserved_segmentation_endpoints.pdf) · [CSV](results/conference_figures/data/fig11_reserved_segmentation_endpoints.csv)

![Figure 11: Reserved evaluation: four selected model families](results/conference_figures/fig11_reserved_segmentation_endpoints.png)

**Figure 14. All reserved modality subsets** — [caption and evidence](docs/FIGURE_CATALOG.md) · [PDF](results/conference_figures/fig14_reserved_modality_subsets.pdf) · [CSV](results/conference_figures/data/fig14_reserved_modality_subsets.csv)

![Figure 14: All reserved modality subsets](results/conference_figures/fig14_reserved_modality_subsets.png)

**Figure 18. Historical exposure sensitivity** — [caption and evidence](docs/FIGURE_CATALOG.md) · [PDF](results/conference_figures/fig18_historical_exposure_sensitivity.pdf) · [CSV](results/conference_figures/data/fig18_historical_exposure_sensitivity.csv)

![Figure 18: Historical exposure sensitivity](results/conference_figures/fig18_historical_exposure_sensitivity.png)

**Figure 19. Prespecified qualitative predictions** — [caption and evidence](docs/FIGURE_CATALOG.md) · [PDF](results/conference_figures/fig19_prespecified_qualitative.pdf) · [CSV](results/conference_figures/data/fig19_prespecified_qualitative.csv)

![Figure 19: Prespecified qualitative predictions](results/conference_figures/fig19_prespecified_qualitative.png)

<details>
<summary><strong>Data audit and separate classification benchmark</strong></summary>

**Figure 01. Dataset audit** — [caption and evidence](docs/FIGURE_CATALOG.md) · [PDF](results/conference_figures/fig01_dataset_audit.pdf) · [CSV](results/conference_figures/data/fig01_dataset_audit.csv)

![Figure 01: Dataset audit](results/conference_figures/fig01_dataset_audit.png)

**Figure 03. Classification ablation summary** — [caption and evidence](docs/FIGURE_CATALOG.md) · [PDF](results/conference_figures/fig03_classification_ablations.pdf) · [CSV](results/conference_figures/data/fig03_classification_ablations.csv)

![Figure 03: Classification ablation summary](results/conference_figures/fig03_classification_ablations.png)

**Figure 04. Classification ROC and precision–recall** — [caption and evidence](docs/FIGURE_CATALOG.md) · [PDF](results/conference_figures/fig04_classification_roc_pr.pdf) · [CSV](results/conference_figures/data/fig04_classification_roc_pr.csv)

![Figure 04: Classification ROC and precision–recall](results/conference_figures/fig04_classification_roc_pr.png)

**Figure 05. Classification confusion matrices** — [caption and evidence](docs/FIGURE_CATALOG.md) · [PDF](results/conference_figures/fig05_classification_confusion.pdf) · [CSV](results/conference_figures/data/fig05_classification_confusion.csv)

![Figure 05: Classification confusion matrices](results/conference_figures/fig05_classification_confusion.png)

**Figure 06. Paired classification contrasts** — [caption and evidence](docs/FIGURE_CATALOG.md) · [PDF](results/conference_figures/fig06_classification_paired_contrasts.pdf) · [CSV](results/conference_figures/data/fig06_classification_paired_contrasts.csv)

![Figure 06: Paired classification contrasts](results/conference_figures/fig06_classification_paired_contrasts.png)

</details>

<details>
<summary><strong>Development results and optimization diagnostics</strong></summary>

**Figure 07. Main segmentation: all nine variants** — [caption and evidence](docs/FIGURE_CATALOG.md) · [PDF](results/conference_figures/fig07_main_segmentation_endpoints.pdf) · [CSV](results/conference_figures/data/fig07_main_segmentation_endpoints.csv)

![Figure 07: Main segmentation: all nine variants](results/conference_figures/fig07_main_segmentation_endpoints.png)

**Figure 09. Training selection curves** — [caption and evidence](docs/FIGURE_CATALOG.md) · [PDF](results/conference_figures/fig09_training_selection_curves.pdf) · [CSV](results/conference_figures/data/fig09_training_selection_curves.csv)

![Figure 09: Training selection curves](results/conference_figures/fig09_training_selection_curves.png)

**Figure 10. Segmentation capacity and compute** — [caption and evidence](docs/FIGURE_CATALOG.md) · [PDF](results/conference_figures/fig10_segmentation_compute.pdf) · [CSV](results/conference_figures/data/fig10_segmentation_compute.csv)

![Figure 10: Segmentation capacity and compute](results/conference_figures/fig10_segmentation_compute.png)

**Figure 20. Lossless cache throughput** — [caption and evidence](docs/FIGURE_CATALOG.md) · [PDF](results/conference_figures/fig20_cache_throughput.pdf) · [CSV](results/conference_figures/data/fig20_cache_throughput.csv)

![Figure 20: Lossless cache throughput](results/conference_figures/fig20_cache_throughput.png)

</details>

<details>
<summary><strong>Additional reserved comparisons, attribution and failures</strong></summary>

**Figure 12. Reserved paired comparisons** — [caption and evidence](docs/FIGURE_CATALOG.md) · [PDF](results/conference_figures/fig12_reserved_paired_contrasts.pdf) · [CSV](results/conference_figures/data/fig12_reserved_paired_contrasts.csv)

![Figure 12: Reserved paired comparisons](results/conference_figures/fig12_reserved_paired_contrasts.png)

**Figure 13. Reserved regional Dice** — [caption and evidence](docs/FIGURE_CATALOG.md) · [PDF](results/conference_figures/fig13_reserved_regional_dice.pdf) · [CSV](results/conference_figures/data/fig13_reserved_regional_dice.csv)

![Figure 13: Reserved regional Dice](results/conference_figures/fig13_reserved_regional_dice.png)

**Figure 15. Reserved modality Shapley contributions** — [caption and evidence](docs/FIGURE_CATALOG.md) · [PDF](results/conference_figures/fig15_reserved_modality_shapley.pdf) · [CSV](results/conference_figures/data/fig15_reserved_modality_shapley.csv)

![Figure 15: Reserved modality Shapley contributions](results/conference_figures/fig15_reserved_modality_shapley.png)

**Figure 16. Reserved group heterogeneity** — [caption and evidence](docs/FIGURE_CATALOG.md) · [PDF](results/conference_figures/fig16_reserved_group_heterogeneity.pdf) · [CSV](results/conference_figures/data/fig16_reserved_group_heterogeneity.csv)

![Figure 16: Reserved group heterogeneity](results/conference_figures/fig16_reserved_group_heterogeneity.png)

**Figure 17. Empty-reference ET failures** — [caption and evidence](docs/FIGURE_CATALOG.md) · [PDF](results/conference_figures/fig17_empty_reference_et_failures.pdf) · [CSV](results/conference_figures/data/fig17_empty_reference_et_failures.csv)

![Figure 17: Empty-reference ET failures](results/conference_figures/fig17_empty_reference_et_failures.png)

</details>

## Limits that belong in the paper

- Similarity groups are not verified patients. The repaired grouping addresses observed copies, not every possible patient relationship.
- Seven reserved groups include earlier notebook validation cases. The 15-group historical sensitivity is exploratory and overlaps the complete cohort.
- Missing modalities are masked after complete-modality preprocessing; this does not validate an acquisition workflow that never obtained those sequences.
- Main training uses one development fold and three fixed seeds. Group intervals and seed SD are different quantities.
- Cross-family comparisons include auxiliary-objective differences. Same-family controls provide closer tests of the quality contribution.
- Both ET-empty cases receive false-positive ET from every model and seed. Figure 17 retains those failures.

## Notebooks and reproducibility

| Notebook | Purpose |
|---|---|
| [01 — classification ablations](notebooks/01_classification_ablation.ipynb) | Original completed 105-fit research source |
| [02 — segmentation study](notebooks/02_segmentation_study.ipynb) | Original completed main-study research source |
| [03 — conference visualizations](notebooks/03_conference_visualizations.ipynb) | Supporting Kaggle CPU report, with private evidence-dataset attachment |
| [Executed visualization gallery](results/conference_figures/03_conference_visualizations.executed.ipynb) | Actual Kaggle output notebook with all 20 figures and captions |

Install the declared dependencies in a suitable Python environment. Rebuild figures into a separate local output directory to preserve the verified Kaggle exports:

```bash
python -m pip install -r requirements-dev.txt
PYTHONPATH=segmentation/src python -m pytest tests segmentation/tests -q
python scripts/build_conference_figures.py --output runtime/figure_rebuild
python scripts/build_paper_tables.py
```

The original verification suite contains 186 tests. Figure validation additionally checks source hashes, complete model/seed/cohort coverage, plotted values and actual notebook image bytes. Detailed execution instructions are in the [notebook index](notebooks/README.md) and [operation guide](docs/NOTEBOOKS.md).

## Repository and access

Source lives in `classification/` and `segmentation/`; `notebooks/` contains portable notebook sources; `kaggle/` indexes deployment bundles; `results/` contains verified evidence; `docs/` contains Markdown, DOCX and PDF reports; `paper/` contains writing tables, references and the claim map; `audit/` preserves provenance.

The GitHub repository and Kaggle jobs are private. Sign in to the owning account or obtain the appropriate collaborator access. Credentials, raw MRI volumes, image datasets and checkpoints are excluded from Git history. Dataset/checkpoint access remains documented through Kaggle. Preserve [the montage attribution](results/conference_figures/ATTRIBUTION.md) with its figures; it does not grant a license to the entire codebase.

[Download v1.1.0 writing package](https://github.com/Hisernberg/mri-conference-research/releases/tag/v1.1.0) · [Preserved v1.0.0 experiment release](https://github.com/Hisernberg/mri-conference-research/releases/tag/v1.0.0) · [GitHub release guide](docs/GITHUB_RELEASE.md)
