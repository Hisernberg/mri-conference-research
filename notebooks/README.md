# Research notebooks

The two notebooks below contain the reviewed source snapshots and frozen execution settings. Their outputs are cleared for readable GitHub rendering; measured outcomes are preserved in the result tables, verification records and linked Kaggle executions.

| Notebook | Purpose | Completed Kaggle run |
|---|---|---|
| [01 — Classification ablations](01_classification_ablation.ipynb) | Seven variants, five grouped outer folds and three seeds: 105 fits | [Classification v2, version 2](https://www.kaggle.com/code/dasshovon/mri-classification-ablation-v2) |
| [02 — Segmentation study](02_segmentation_study.ipynb) | Nine variants, three seeds, grouped development fold 0; the checked-in execution cell selects seed 42 | [Main seed 42, version 1](https://www.kaggle.com/code/dasshovon/mri-segmentation-study-s42) |

The segmentation matrix also includes [seed 43](https://www.kaggle.com/code/dasshovon/mri-segmentation-study-s43) and [seed 44](https://www.kaggle.com/code/dasshovon/mri-segmentation-study-s44), both version 1 with the same embedded library source hash. Their exact submitted execution cells and source dictionaries are retained in [the source snapshots](../audit/source_snapshots/).

Use the [operation guide](../docs/NOTEBOOKS.md) and its [PDF](../docs/pdf/NOTEBOOKS.pdf) for dataset attachments, preparation, seed-specific generation, checkpoint selection and independent result verification. The [Kaggle run index](../kaggle/README.md) includes the preparation, pilot, optimization and reserved-evaluation jobs.

Kaggle links point to private notebooks/scripts. Select the stated version in Kaggle's version history. Copying or running a notebook launches a new experiment; the files and links alone do not execute training. No credential belongs in a notebook cell. Training used frozen internal image groups; patient identity and external clinical generalization are unverified.

## Supporting conference visualization notebook

[03_conference_visualizations.ipynb](03_conference_visualizations.ipynb) is a separate CPU reporting notebook. It renders 20 figures from the frozen result tables and preserves the original real-prediction montage. It does not change the two research notebooks or train/evaluate another model.

- [Kaggle execution, version 1](https://www.kaggle.com/code/dasshovon/mri-conference-20-visualizations)
- [Actually executed notebook with 20 image outputs](../results/conference_figures/03_conference_visualizations.executed.ipynb)
- [Complete figure catalog](../docs/FIGURE_CATALOG.md) and [20-page atlas](../results/conference_figures/conference_figure_atlas.pdf)
- [Independent verification](../audit/conference_visualizations_verification.json)

The input attachment is the private [figure-evidence dataset, version 1](https://www.kaggle.com/datasets/dasshovon/mri-conference-figure-evidence). The notebook validates its manifest and every input hash before rendering. This small derived bundle was separated from notebook source to respect Kaggle's 1 MB source limit. Its separate source/figure terms are preserved; it is not a new MRI cohort.

From a Git clone containing tag v1.0.0, `python scripts/build_visualization_notebook.py` rebuilds the portable source and prepares the local input bundle under ignored runtime storage. For a direct local figure rebuild, use `python scripts/build_conference_figures.py --output runtime/figure_rebuild`. Keep the downloaded exports and their execution receipt intact. The readable plotting source is in `scripts/build_conference_figures.py`.
