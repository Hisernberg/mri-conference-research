# Audited MRI experiments

Two completed Kaggle studies derived from the supplied notebooks: an image-level brain MRI classification benchmark and a corrected missing-modality segmentation project. The segmentation study provides the main research direction, with explicit ablations and negative findings. These internal benchmarks do not establish patient-level external validation, architectural novelty or conference acceptance.

[Research notebooks](notebooks/README.md) · [Kaggle runs and versions](kaggle/README.md) · [Result files](results/README.md) · [Download release](https://github.com/Hisernberg/mri-conference-research/releases/tag/v1.0.0)

Repository: [Hisernberg/mri-conference-research](https://github.com/Hisernberg/mri-conference-research). The initial GitHub publication is private. Kaggle executions are also private and require access to the `dasshovon` account. The release archive includes the reviewed code, both notebooks, all reports and curated evidence; dataset and checkpoint access are documented through Kaggle.

- [Experiment plan](EXPERIMENT_PLAN.md) · [PDF](docs/pdf/EXPERIMENT_PLAN.pdf)
- [Notebook audit and historical evidence](docs/NOTEBOOK_AUDIT.md) · [PDF](docs/pdf/NOTEBOOK_AUDIT.pdf)
- [Datasets and split limitations](docs/DATASETS.md) · [PDF](docs/pdf/DATASETS.pdf)
- [Notebook operation](docs/NOTEBOOKS.md) · [PDF](docs/pdf/NOTEBOOKS.pdf)
- [Actual execution and results](docs/RESULTS.md) · [PDF](docs/pdf/RESULTS.pdf)
- [Related work and conference positioning](docs/RELATED_WORK.md) · [PDF](docs/pdf/RELATED_WORK.pdf)
- [GitHub release guide](docs/GITHUB_RELEASE.md) · [PDF](docs/pdf/GITHUB_RELEASE.pdf)

Research notebooks: [classification ablations](notebooks/01_classification_ablation.ipynb) and [segmentation study](notebooks/02_segmentation_study.ipynb). Source is maintained in `classification/` and `segmentation/src/qmmf/`; the generated source snapshot makes each notebook portable to Kaggle. A supporting private CPU script prepares segmentation caches without using GPU time.

The dataset audit independently confirms 506 classification files, 228 unique images and 209 conservative similarity groups. Patient identities are unavailable. The segmentation repairs verify channel order, preserve inference geometry, apply training-fitted quality normalization, provide genuine full-modality teacher inputs and activate the shuffled-quality control.

The completed Kaggle classification matrix contains 105 runs. The full model has mean OOF macro F1 **0.7745**, accuracy **0.7807**, and AUROC **0.8457** across three seeds. Its F1 advantage over the narrow ResNet is uncertain under descriptive paired group bootstrap. Segmentation audits found rescaled image copies crossing the original case-ID splits. The initial pilot was cancelled; repaired partitions isolate 262 conservative image groups, reserve 51 groups for the revised internal evaluation, and weight groups equally.

A retrospective audit identified seven reserved groups that include cases from the supplied notebook's earlier validation subset. Thus protected refers to the revised experiment's fitting and selection boundary, not a historically untouched cohort. The complete 51-group report is retained. An additional completed exploratory sensitivity analysis, frozen before revised reserved-cohort outcomes, uses 15 groups/18 cases entirely outside the original development partition. Historical use, unknown patient identity and the small sensitivity cohort limit inference; details are in the dataset and results documents.

The corrected short segmentation pilot is complete, with no established quality-module benefit. Verified lossless cache changes measured a **4.02×** QMMF training-segment speedup on Kaggle. The frozen main study contains **nine variants × three seeds**, up to **120 epochs × 100 microbatches** per fit.

As of 2026-09-09 11:13 UTC, **all 27 main fits and all 12 reserved checkpoint evaluations are complete and independently verified**. Main development Dice is **0.8052 +/- 0.0028** for QMMF and **0.8106 +/- 0.0042** without quality features. The capacity-matched moment control has a better mean worst-subset score under the declared paired group adjustment.

On the reserved 66 cases/51 groups, QMMF full-modality Dice is **0.7689 +/- 0.0055**, versus **0.7769 +/- 0.0007** without quality features. QMMF minus no quality is **-0.0080**, adjusted group interval **[-0.0133, -0.0028]**; mean- and worst-subset comparisons also favor no quality. The 15-group historical sensitivity leaves the full-modality difference uncertain and still favors no quality on worst-subset Dice. These results do not support the proposed quality-conditioning advantage. Both ET-empty reserved cases receive false-positive ET predictions from every model and seed; the report retains this failure alongside all models, intervals, regional tables and three fixed qualitative examples.

Run local verification:

```bash
PYTHONPATH=segmentation/src python -m pytest tests segmentation/tests -q
```

The suite uses synthetic fixtures for code correctness. Scientific results must be read from completed Kaggle artifacts. See the results document for current run status, measured scores, failures and limitations.

Credentials, full MRI volumes, caches and classification images are excluded from this repository. Any declared derived MSD single-slice figures retain source attribution and CC BY-SA 4.0 terms. The source-code license requires author confirmation; dataset terms remain those of the original sources.
