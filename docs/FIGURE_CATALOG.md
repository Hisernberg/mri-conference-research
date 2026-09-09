# Conference figure catalog

All 20 figures were produced or preserved by [Kaggle CPU notebook version 1](https://www.kaggle.com/code/dasshovon/mri-conference-20-visualizations). The output bundle, plotted values, source files and all 20 image outputs in the executed notebook were independently checked. The original MRI montage's PNG/PDF bytes are unchanged.

[Executed notebook](../results/conference_figures/03_conference_visualizations.executed.ipynb) · [20-page PDF atlas](../results/conference_figures/conference_figure_atlas.pdf) · [Source/output hashes](../results/conference_figures/figure_manifest.json) · [Kaggle verification](../audit/conference_visualizations_verification.json)

The atlas is assembled locally from the downloaded PDF pages without replotting. This catalog supplies complete captions and interpretation limits. Chart PDF/SVG files have vector elements; Figure 19 is a preserved raster montage with [CC BY-SA 4.0 attribution](../results/conference_figures/ATTRIBUTION.md). Figure generation adds no training, inference or new hypothesis tests.

| ID | Figure | Writing role |
|---|---|---|
| 01 | Dataset audit | Methods / data audit |
| 02 | Frozen segmentation cohorts | Methods / evaluation design |
| 03 | Classification ablation summary | Supplement / separate classification benchmark |
| 04 | Classification ROC and precision–recall | Supplement / classifier discrimination |
| 05 | Classification confusion matrices | Supplement / classifier error analysis |
| 06 | Paired classification contrasts | Supplement / classifier ablations |
| 07 | Main segmentation: all nine variants | Main paper / main ablations |
| 08 | Main paired ablation intervals | Main paper / ablation evidence |
| 09 | Training selection curves | Supplement / optimization diagnostics |
| 10 | Segmentation capacity and compute | Supplement / resources |
| 11 | Reserved evaluation: four selected model families | Main paper / primary results |
| 12 | Reserved paired comparisons | Main paper / reserved comparisons |
| 13 | Reserved regional Dice | Main paper / regional results |
| 14 | All reserved modality subsets | Main paper / missing-modality robustness |
| 15 | Reserved modality Shapley contributions | Supplement / modality attribution |
| 16 | Reserved group heterogeneity | Supplement / distribution of errors |
| 17 | Empty-reference ET failures | Main paper / failure analysis |
| 18 | Historical exposure sensitivity | Main paper / limitations and sensitivity |
| 19 | Prespecified qualitative predictions | Main paper / qualitative examples |
| 20 | Lossless cache throughput | Supplement / implementation efficiency |

## Figure 01. Dataset audit

![Dataset audit](../results/conference_figures/fig01_dataset_audit.png)

Classification auditing reduced 506 files to 228 decoded unique images in 209 conservative similarity groups. The segmentation dataset contains 484 canonical case IDs in 262 conservative groups after the rescaled-copy audit. Counts describe different units and must not be interpreted as patient counts.

**Interpretation:** Image grouping addresses observed duplication; it cannot establish patient independence.

[PNG](../results/conference_figures/fig01_dataset_audit.png) · [PDF](../results/conference_figures/fig01_dataset_audit.pdf) · [SVG](../results/conference_figures/fig01_dataset_audit.svg) · [Plotted CSV](../results/conference_figures/data/fig01_dataset_audit.csv)

Sources: [dataset_audit.json](../results/classification/audit/dataset_audit.json), [image_manifest.csv](../results/classification/audit/image_manifest.csv), [case_groups.json](../results/segmentation_audit/grouped_v2/case_groups.json).


## Figure 02. Frozen segmentation cohorts

![Frozen segmentation cohorts](../results/conference_figures/fig02_segmentation_cohorts.png)

The executed fold-0 partition contains 294 training cases/132 groups, 33 inner-selection cases/24 groups, 91 outer-development cases/55 groups, and 66 reserved cases/51 groups. All 27 main fits share this partition. Missing-modality development evaluation uses 16 fixed group representatives; reserved subset evaluation uses 51 representatives. The 15-group historical sensitivity is nested within the reserved set.

**Interpretation:** Four-fold metadata exists, but the reported main experiment is one grouped development fold repeated across three seeds.

[PNG](../results/conference_figures/fig02_segmentation_cohorts.png) · [PDF](../results/conference_figures/fig02_segmentation_cohorts.pdf) · [SVG](../results/conference_figures/fig02_segmentation_cohorts.svg) · [Plotted CSV](../results/conference_figures/data/fig02_segmentation_cohorts.csv)

Sources: [split_counts.csv](../results/segmentation_audit/grouped_v2/split_counts.csv), [verification.json](../results/segmentation_protected/verification.json).


## Figure 03. Classification ablation summary

![Classification ablation summary](../results/conference_figures/fig03_classification_ablations.png)

Out-of-fold macro F1 and AUROC are summarized over the three fixed seeds for every classifier variant. Points show the seed mean and bars show one sample standard deviation across seeds. Each seed contributes one out-of-fold prediction per unique image. These bars are not confidence intervals.

**Interpretation:** The full classifier has mean macro F1 0.7745; its paired advantage over the narrow ResNet remains uncertain.

[PNG](../results/conference_figures/fig03_classification_ablations.png) · [PDF](../results/conference_figures/fig03_classification_ablations.pdf) · [SVG](../results/conference_figures/fig03_classification_ablations.svg) · [Plotted CSV](../results/conference_figures/data/fig03_classification_ablations.csv)

Sources: [model_summary.csv](../results/classification/model_summary.csv).


## Figure 04. Classification ROC and precision–recall

![Classification ROC and precision–recall](../results/conference_figures/fig04_classification_roc_pr.png)

ROC and precision–recall curves use the original out-of-fold probabilities for each of three seeds. Each curve contains 228 unique images; repeated seeds are displayed separately and are not pooled as independent observations. The ROC diagonal and positive-class prevalence are reference lines. Legend values are the saved AUROC and average precision.

**Interpretation:** These are internal, descriptive image-level discrimination curves, not calibrated clinical operating points.

[PNG](../results/conference_figures/fig04_classification_roc_pr.png) · [PDF](../results/conference_figures/fig04_classification_roc_pr.pdf) · [SVG](../results/conference_figures/fig04_classification_roc_pr.svg) · [Plotted CSV](../results/conference_figures/data/fig04_classification_roc_pr.csv)

Sources: [oof_predictions.csv](../results/classification/oof_predictions.csv), [per_seed_oof_metrics.csv](../results/classification/per_seed_oof_metrics.csv).


## Figure 05. Classification confusion matrices

![Classification confusion matrices](../results/conference_figures/fig05_classification_confusion.png)

Out-of-fold confusion matrices for the full classifier at the prespecified probability threshold of 0.5. Every seed uses the same 228 unique images: 87 negative and 141 positive. The image counts cannot be interpreted as independent patients, and the three matrices must not be summed to inflate sample size.

**Interpretation:** Both false negatives and false positives remain visible for all fixed seeds.

[PNG](../results/conference_figures/fig05_classification_confusion.png) · [PDF](../results/conference_figures/fig05_classification_confusion.pdf) · [SVG](../results/conference_figures/fig05_classification_confusion.svg) · [Plotted CSV](../results/conference_figures/data/fig05_classification_confusion.csv)

Sources: [oof_predictions.csv](../results/classification/oof_predictions.csv).


## Figure 06. Paired classification contrasts

![Paired classification contrasts](../results/conference_figures/fig06_classification_paired_contrasts.png)

Saved paired contrasts compare the full classifier with all six controls. Points are the mean seed-level OOF macro-F1 differences; bars are descriptive percentile intervals from group resampling with fixed seeds retained. These follow-up intervals are unadjusted for multiplicity. The narrow-ResNet interval crosses zero.

**Interpretation:** Use this as exploratory supporting evidence; do not present all positive intervals as confirmatory discoveries.

[PNG](../results/conference_figures/fig06_classification_paired_contrasts.png) · [PDF](../results/conference_figures/fig06_classification_paired_contrasts.pdf) · [SVG](../results/conference_figures/fig06_classification_paired_contrasts.svg) · [Plotted CSV](../results/conference_figures/data/fig06_classification_paired_contrasts.csv)

Sources: [paired_ablation_deltas.csv](../results/classification/paired_ablation_deltas.csv).


## Figure 07. Main segmentation: all nine variants

![Main segmentation: all nine variants](../results/conference_figures/fig07_main_segmentation_endpoints.png)

Model comparison across full-modality, mean-subset and worst-subset Dice. The cohort comprises 55 outer-development groups (full modalities) and 16 fixed representatives (subsets). Filled points show means across the three fixed seeds; small ticks show individual seed values. Horizontal bars show saved 95% group-bootstrap intervals retaining all three seeds. Interval width is group uncertainty, not seed SD. The mean- and worst-subset endpoints use all 15 nonempty modality subsets.

**Interpretation:** Quality conditioning does not establish an advantage over the no-quality control; the paired intervals are the appropriate comparisons.

[PNG](../results/conference_figures/fig07_main_segmentation_endpoints.png) · [PDF](../results/conference_figures/fig07_main_segmentation_endpoints.pdf) · [SVG](../results/conference_figures/fig07_main_segmentation_endpoints.svg) · [Plotted CSV](../results/conference_figures/data/fig07_main_segmentation_endpoints.csv)

Sources: [repeated_seed_metrics.csv](../results/segmentation_main/repeated_seed_metrics.csv), [seed_specific_metrics.csv](../results/segmentation_main/seed_specific_metrics.csv).


## Figure 08. Main paired ablation intervals

![Main paired ablation intervals](../results/conference_figures/fig08_main_paired_ablations.png)

All declared QMMF-control contrasts for the three segmentation endpoints. Bars are the saved Bonferroni-adjusted paired group-bootstrap intervals, using a family of 8 controls separately for each endpoint and retaining three fixed training seeds. Negative values favor the control. The intervals are reproduced from the frozen analysis; no new comparisons were selected for this figure.

**Interpretation:** The main development comparisons do not support the quality module; all reserved no-quality contrasts favor removing it.

[PNG](../results/conference_figures/fig08_main_paired_ablations.png) · [PDF](../results/conference_figures/fig08_main_paired_ablations.pdf) · [SVG](../results/conference_figures/fig08_main_paired_ablations.svg) · [Plotted CSV](../results/conference_figures/data/fig08_main_paired_ablations.csv)

Sources: [paired_repeated_seed_deltas.csv](../results/segmentation_main/paired_repeated_seed_deltas.csv).


## Figure 09. Training selection curves

![Training selection curves](../results/conference_figures/fig09_training_selection_curves.png)

Inner-validation selection scores at all 12 scheduled checks for every model and seed. Each fit ran 120 epochs; dots indicate the checkpoint selected by the original inner-validation rule. Lines connect observed checks without smoothing or extrapolation. The vertical score is the recorded composite selection objective, not the outer or reserved test score.

**Interpretation:** The fixed budget and available validation trajectory do not establish convergence for every model.

[PNG](../results/conference_figures/fig09_training_selection_curves.png) · [PDF](../results/conference_figures/fig09_training_selection_curves.pdf) · [SVG](../results/conference_figures/fig09_training_selection_curves.svg) · [Plotted CSV](../results/conference_figures/data/fig09_training_selection_curves.csv)

Sources: [all_training_histories.csv](../results/segmentation_main/all_training_histories.csv), [training_diagnostics.csv](../results/segmentation_main/training_diagnostics.csv).


## Figure 10. Segmentation capacity and compute

![Segmentation capacity and compute](../results/conference_figures/fig10_segmentation_compute.png)

Aligned rows show trainable parameters (left), mean outer-development Dice with one seed SD (center), and component-wise median fit/validation and evaluation durations over three seeds (right). Duration components are stacked for readability; their sum is not the median total runtime. Cross-family objectives differ: QMMF and its controls have auxiliary heads while HeMIS-style and U-Net do not.

**Interpretation:** This is an observed implementation trade-off, not a controlled architecture-only efficiency claim or an account billing estimate.

[PNG](../results/conference_figures/fig10_segmentation_compute.png) · [PDF](../results/conference_figures/fig10_segmentation_compute.pdf) · [SVG](../results/conference_figures/fig10_segmentation_compute.svg) · [Plotted CSV](../results/conference_figures/data/fig10_segmentation_compute.csv)

Sources: [repeated_seed_metrics.csv](../results/segmentation_main/repeated_seed_metrics.csv), [resource_summary.csv](../results/segmentation_main/resource_summary.csv).


## Figure 11. Reserved evaluation: four selected model families

![Reserved evaluation: four selected model families](../results/conference_figures/fig11_reserved_segmentation_endpoints.png)

Model comparison across full-modality, mean-subset and worst-subset Dice. The cohort comprises 51 reserved groups: 66 cases for full modalities and 51 representatives for subsets. Filled points show means across the three fixed seeds; small ticks show individual seed values. Horizontal bars show saved 95% group-bootstrap intervals retaining all three seeds. Interval width is group uncertainty, not seed SD. The mean- and worst-subset endpoints use all 15 nonempty modality subsets.

**Interpretation:** Quality conditioning does not establish an advantage over the no-quality control; the paired intervals are the appropriate comparisons.

[PNG](../results/conference_figures/fig11_reserved_segmentation_endpoints.png) · [PDF](../results/conference_figures/fig11_reserved_segmentation_endpoints.pdf) · [SVG](../results/conference_figures/fig11_reserved_segmentation_endpoints.svg) · [Plotted CSV](../results/conference_figures/data/fig11_reserved_segmentation_endpoints.csv)

Sources: [repeated_seed_metrics.csv](../results/segmentation_protected/repeated_seed_metrics.csv), [seed_specific_metrics.csv](../results/segmentation_protected/seed_specific_metrics.csv).


## Figure 12. Reserved paired comparisons

![Reserved paired comparisons](../results/conference_figures/fig12_reserved_paired_contrasts.png)

All declared QMMF-control contrasts for the three segmentation endpoints. Bars are the saved Bonferroni-adjusted paired group-bootstrap intervals, using a family of 3 controls separately for each endpoint and retaining three fixed training seeds. Negative values favor the control. The intervals are reproduced from the frozen analysis; no new comparisons were selected for this figure.

**Interpretation:** The main development comparisons do not support the quality module; all reserved no-quality contrasts favor removing it.

[PNG](../results/conference_figures/fig12_reserved_paired_contrasts.png) · [PDF](../results/conference_figures/fig12_reserved_paired_contrasts.pdf) · [SVG](../results/conference_figures/fig12_reserved_paired_contrasts.svg) · [Plotted CSV](../results/conference_figures/data/fig12_reserved_paired_contrasts.csv)

Sources: [paired_repeated_seed_deltas.csv](../results/segmentation_protected/paired_repeated_seed_deltas.csv).


## Figure 13. Reserved regional Dice

![Reserved regional Dice](../results/conference_figures/fig13_reserved_regional_dice.png)

Group-aggregated Dice for WT, TC and ET in the reserved evaluation. Points show the mean over three seeds and bars show seed SD. The frozen nonempty-reference rule excludes two ET-empty cases from ET Dice; WT and TC have 66 nonempty reference cases, while ET has 64. False-positive ET on the excluded cases is reported separately in Figure 17.

**Interpretation:** Regional averages must be read together with the empty-reference failure analysis.

[PNG](../results/conference_figures/fig13_reserved_regional_dice.png) · [PDF](../results/conference_figures/fig13_reserved_regional_dice.pdf) · [SVG](../results/conference_figures/fig13_reserved_regional_dice.svg) · [Plotted CSV](../results/conference_figures/data/fig13_reserved_regional_dice.csv)

Sources: [regional_summary.csv](../results/segmentation_protected/reporting/regional_summary.csv).


## Figure 14. All reserved modality subsets

![All reserved modality subsets](../results/conference_figures/fig14_reserved_modality_subsets.png)

Heatmap of all 15 nonempty subsets of T1, T1ce, T2 and FLAIR for four models. Values are group-level scores averaged over the three fixed seeds and the 51 reserved representatives. The scale is fixed at 0–1 and each value is shown numerically. Complete four-channel preprocessing occurs before channel masking, so the experiment measures simulated missingness.

**Interpretation:** No quality has the higher descriptive mean than QMMF in all 15 combinations; individual cells are not new adjusted hypothesis tests.

[PNG](../results/conference_figures/fig14_reserved_modality_subsets.png) · [PDF](../results/conference_figures/fig14_reserved_modality_subsets.pdf) · [SVG](../results/conference_figures/fig14_reserved_modality_subsets.svg) · [Plotted CSV](../results/conference_figures/data/fig14_reserved_modality_subsets.csv)

Sources: [modality_subset_summary.csv](../results/segmentation_protected/reporting/modality_subset_summary.csv).


## Figure 15. Reserved modality Shapley contributions

![Reserved modality Shapley contributions](../results/conference_figures/fig15_reserved_modality_shapley.png)

Shapley values are computed from all 15 observed nonempty subset scores and a declared empty-subset value of zero. Group means are averaged across the three seeds; bars show seed SD. Contributions are conditional on the fitted models, the preprocessing and the chosen utility convention, and are not causal importance estimates.

**Interpretation:** T1ce has the largest descriptive contribution in this declared game; no claim about clinical indispensability follows.

[PNG](../results/conference_figures/fig15_reserved_modality_shapley.png) · [PDF](../results/conference_figures/fig15_reserved_modality_shapley.pdf) · [SVG](../results/conference_figures/fig15_reserved_modality_shapley.svg) · [Plotted CSV](../results/conference_figures/data/fig15_reserved_modality_shapley.csv)

Sources: [modality_shapley_summary.csv](../results/segmentation_protected/reporting/modality_shapley_summary.csv).


## Figure 16. Reserved group heterogeneity

![Reserved group heterogeneity](../results/conference_figures/fig16_reserved_group_heterogeneity.png)

Each scatter point is one of 51 conservative reserved image groups after averaging the three fixed seeds. The diagonal denotes equal performance. The histogram displays every corresponding paired group difference and its mean, rather than selecting only favorable or unfavorable examples. It adds no hypothesis test.

**Interpretation:** The aggregate negative difference coexists with group-specific gains and losses; neither plot establishes patient-level effects.

[PNG](../results/conference_figures/fig16_reserved_group_heterogeneity.png) · [PDF](../results/conference_figures/fig16_reserved_group_heterogeneity.pdf) · [SVG](../results/conference_figures/fig16_reserved_group_heterogeneity.svg) · [Plotted CSV](../results/conference_figures/data/fig16_reserved_group_heterogeneity.csv)

Sources: [full_group_macro_dice_seed42_by_group.csv](../results/segmentation_protected/full_group_macro_dice_seed42_by_group.csv), [full_group_macro_dice_seed43_by_group.csv](../results/segmentation_protected/full_group_macro_dice_seed43_by_group.csv), [full_group_macro_dice_seed44_by_group.csv](../results/segmentation_protected/full_group_macro_dice_seed44_by_group.csv).


## Figure 17. Empty-reference ET failures

![Empty-reference ET failures](../results/conference_figures/fig17_empty_reference_et_failures.png)

False-positive ET volume for both reserved cases whose reference ET volume is zero. All four models and all three seeds are shown (24 observations). Every observation has positive predicted ET volume. These cases are excluded from the primary nonempty-reference ET Dice according to the frozen rule, but their failures remain explicitly reported here.

**Interpretation:** The aggregate Dice results do not imply reliable rejection of absent enhancing tumor.

[PNG](../results/conference_figures/fig17_empty_reference_et_failures.png) · [PDF](../results/conference_figures/fig17_empty_reference_et_failures.pdf) · [SVG](../results/conference_figures/fig17_empty_reference_et_failures.svg) · [Plotted CSV](../results/conference_figures/data/fig17_empty_reference_et_failures.csv)

Sources: [all_full_case_metrics.csv](../results/segmentation_protected/all_full_case_metrics.csv).


## Figure 18. Historical exposure sensitivity

![Historical exposure sensitivity](../results/conference_figures/fig18_historical_exposure_sensitivity.png)

QMMF minus no-quality differences for the complete 51-group reserved cohort and the frozen 15-group/18-case sensitivity whose groups are entirely outside the original notebook's development partition. Bars reproduce adjusted intervals (three controls per endpoint). The sensitivity was frozen before reserved outcomes but after some main development results were available; it remains exploratory.

**Interpretation:** The smaller sensitivity leaves full and mean-subset differences uncertain while still favoring no quality on worst-subset Dice. It does not replace the complete reserved report.

[PNG](../results/conference_figures/fig18_historical_exposure_sensitivity.png) · [PDF](../results/conference_figures/fig18_historical_exposure_sensitivity.pdf) · [SVG](../results/conference_figures/fig18_historical_exposure_sensitivity.svg) · [Plotted CSV](../results/conference_figures/data/fig18_historical_exposure_sensitivity.csv)

Sources: [paired_repeated_seed_deltas.csv](../results/segmentation_historical_sensitivity/paired_repeated_seed_deltas.csv), [verification.json](../results/segmentation_historical_sensitivity/verification.json), [paired_repeated_seed_deltas.csv](../results/segmentation_protected/paired_repeated_seed_deltas.csv).


## Figure 19. Prespecified qualitative predictions

![Prespecified qualitative predictions](../results/conference_figures/fig19_prespecified_qualitative.png)

Unmodified, verified prediction montage for the three cases fixed before reserved evaluation: BRATS_143 (slice 77), BRATS_119 (slice 87) and BRATS_023 (slice 77), with all four models at seed 42. Titles report whole-volume macro Dice, not slice Dice. BRATS_143 appeared in the original notebook's validation subset. The figure contains real Kaggle predictions and is reused byte-for-byte in PNG and PDF; the SVG is a raster wrapper.

**Interpretation:** Illustrative examples are prespecified and retain failures and historical exposure. They do not estimate population performance.

[PNG](../results/conference_figures/fig19_prespecified_qualitative.png) · [PDF](../results/conference_figures/fig19_prespecified_qualitative.pdf) · [SVG](../results/conference_figures/fig19_prespecified_qualitative.svg) · [Plotted CSV](../results/conference_figures/data/fig19_prespecified_qualitative.csv)

Sources: [ATTRIBUTION.md](../results/segmentation_protected/qualitative/ATTRIBUTION.md), [panel_metrics.json](../results/segmentation_protected/qualitative/panel_metrics.json), [prespecified_examples.pdf](../results/segmentation_protected/qualitative/prespecified_examples.pdf), [prespecified_examples.png](../results/segmentation_protected/qualitative/prespecified_examples.png), [verification.json](../results/segmentation_protected/qualitative/verification.json).


## Figure 20. Lossless cache throughput

![Lossless cache throughput](../results/conference_figures/fig20_cache_throughput.png)

Measured training-segment throughput and latency for original volume caches versus verified voxel-exact slab caches. Each model/format used 10 warmup and 100 timed microbatches with batch size four and accumulation two. The measured throughput ratios are approximately 4.02× for QMMF and 3.51× for HeMIS-style. These are single bounded measurements, not repeated-run confidence estimates.

**Interpretation:** The speedups apply to the timed training segment and must not be described as end-to-end experiment speedups.

[PNG](../results/conference_figures/fig20_cache_throughput.png) · [PDF](../results/conference_figures/fig20_cache_throughput.pdf) · [SVG](../results/conference_figures/fig20_cache_throughput.svg) · [Plotted CSV](../results/conference_figures/data/fig20_cache_throughput.csv)

Sources: [completion.json](../results/optimization/kaggle_training_io/completion.json), [timing_comparison.csv](../results/optimization/kaggle_training_io/timing_comparison.csv).
