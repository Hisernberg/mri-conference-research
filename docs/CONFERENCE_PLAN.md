# Conference writing and visualization plan

This extension turns the completed research release into a writing package. The experiment matrices, checkpoint choices, frozen protocols and reported scores remain those of release v1.0.0. No additional model training, inference, test-set tuning or hypothesis selection is part of this extension. A separate Kaggle CPU notebook renders the figure collection from the frozen evidence.

The venue and page limit have not been specified. Use a venue-neutral manuscript structure and select a compact main-paper figure set; retain the full 20-figure collection as supporting material. The principal study concerns missing-modality segmentation. The small classification benchmark is separate supporting work and must not be merged into one headline performance claim.

| Figure | Evidence and purpose | Initial writing placement |
|---|---|---|
| 01 | Dataset counts before and after similarity grouping | Data audit |
| 02 | Executed fold-0 roles and reserved cohort | Evaluation design |
| 03 | All seven classification variants | Classification supplement |
| 04 | Per-seed OOF ROC and precision–recall curves | Classification supplement |
| 05 | Three confusion matrices at the frozen threshold | Classification supplement |
| 06 | All six paired classifier contrasts | Classification supplement |
| 07 | All nine main segmentation variants and three endpoints | Main ablation study |
| 08 | Adjusted intervals for all main QMMF-control contrasts | Main ablation study |
| 09 | All 27 inner-selection learning curves | Optimization supplement |
| 10 | Capacity, score and recorded compute | Resource supplement |
| 11 | All four reserved model families and three endpoints | Primary results |
| 12 | All declared reserved paired contrasts | Primary comparisons |
| 13 | WT, TC and ET performance with the empty-reference rule | Regional results |
| 14 | All 15 modality combinations for all four models | Robustness |
| 15 | Modality Shapley values under the declared empty utility | Exploratory attribution |
| 16 | All 51 paired group scores and their differences | Error distribution |
| 17 | Both ET-empty cases, all models and seeds | Failure analysis |
| 18 | Complete cohort and frozen historical sensitivity | Sensitivity / limitations |
| 19 | Unmodified prespecified real prediction montage | Qualitative results |
| 20 | Measured lossless-cache throughput | Implementation supplement |

Each figure receives PNG, PDF and SVG files, a plotted-data CSV, a full caption, a limited interpretation and source-file hashes. Figure 19 preserves the original PNG/PDF bytes and attribution; its SVG is a wrapper around the original raster. Other chart PDF/SVG exports preserve vector elements. The executed Kaggle gallery is retained separately from the portable notebook source.

The paper guide and draft must retain the observed quality-module failure, unknown patient identity, historical validation exposure, simulated missingness after complete-modality preprocessing, fixed training seeds, within-endpoint multiplicity adjustment, auxiliary-objective differences between model families, and lack of independent external validation. Figure count is not a measure of scientific novelty or conference acceptance.

Validation covers source hashes, plotted values, complete seeds/models/groups, image and PDF rendering, readable labels, real Kaggle execution, notebook output images, relative GitHub links and downloaded release checksums. Publication uses a new versioned GitHub release and preserves v1.0.0.
