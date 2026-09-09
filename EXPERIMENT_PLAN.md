# MRI experiment plan

Protocol draft: 2026-09-09. Written before any new training or evaluation.

## Research direction

The supplied notebooks represent two different studies. `brain-mri-01 (1) (1).ipynb` performs binary classification of small, public 2D MRI images. `buet-1.ipynb` embeds QMMF-Net, a missing-modality brain tumour segmentation project for MSD Task01. Their samples, labels, metrics and results must remain separate.

The stronger conference direction is **reliable, efficient segmentation with incomplete MRI modalities**, provided corrected experiments support it. The classification study is an auxiliary reproducibility and ablation benchmark; its missing patient identities prevent a patient-level generalization claim. Neither architectural novelty nor acceptance is established by this plan.

## Stage 1 — audit and repair

1. Preserve and hash both input notebooks. Extract the embedded segmentation source for review and maintainable GitHub publication.
2. Verify Kaggle authentication, exact dataset identifiers, versions, file counts, modality metadata, labels and distribution terms. Keep credentials outside the project. Use private Kaggle notebooks.
3. Classification: check byte and decoded-image duplicates, conflicting labels, approximate duplicate candidates, and filename collisions. Use image/similarity groups, never fabricated patient IDs. Save the complete split manifest.
4. Segmentation: verify channel order against `dataset.json`; pass training-fitted quality normalization through validation and every inference path; ensure the EMA teacher receives the actual full-modality input; implement and test the shuffled-quality control. Investigate non-finite mixed-precision outputs and the saved zero validation scores.
5. Check checkpoint selection, split isolation, resume behavior, parameter comparisons, reconstruction geometry and empty-target metrics. Test changed scientific invariants before launch.

## Stage 2 — two runnable notebooks

| Notebook | Purpose | Evidence |
|---|---|---|
| `01_classification_ablation.ipynb` | Reproducible 2D benchmark with deduplication and component ablations | Grouped out-of-fold predictions, repeated seeds, metrics, confidence intervals, resource measurements |
| `02_segmentation_study.ipynb` | Corrected QMMF-Net training and missing-modality evaluation | Case splits, normalized quality features, baselines/ablations, checkpoints, per-case Dice and robustness tables |

Source modules remain readable outside the notebooks. Notebook generation packages the exact source snapshot for Kaggle and records its hash. No synthetic result may enter a real-data table.

## Classification protocol

- Dataset: `navoneel/brain-mri-images-for-brain-tumor-detection`; confirm current version and deduplicated counts.
- Unit: a unique image or conservatively grouped visually similar images. Similarity is not proof of patient identity. Quarantine cross-label exact duplicates.
- Five outer folds with an inner validation split; identical frozen partitions for all variants. Three initialization seeds if the measured budget permits. A reduced run is explicitly a pilot.
- Primary metric: macro F1. Secondary: balanced accuracy, accuracy, sensitivity, specificity, AUROC, average precision, Brier score and log loss.
- Factorial ablations: attention on/off and multiscale on/off. Additional controls: augmentation off, plain CNN, narrow GroupNorm ResNet. Identify the latter as a custom architecture rather than standard ImageNet ResNet-18.
- Same image resolution, training schedule and stopping criterion across matched variants. Record parameter counts, selected epoch and wall time. No selecting the best outer fold for figures.
- Export every out-of-fold probability with image/group ID, model, fold and seed. Bootstrap independent similarity groups for descriptive uncertainty; overlapping CV fits and unknown patients limit inferential claims. Seeds are not independent patients. Report all runs and negative results.

## Segmentation protocol

- Dataset: MSD Task01 BrainTumour, using an accessible Kaggle mirror. The original notebook reports 484 labeled cases and a 96-case locked test. Confirm on the attached data; unlabeled challenge cases are not an evaluation set.
- Canonical channels: T1, T1ce, T2, FLAIR. Read and reorder the source channels from metadata. Regions: WT, TC, ET derived from verified label semantics.
- Freeze case-level development/test split before training. Development-only checkpoint selection and tuning. Do not inspect the locked test during debugging or pilot iterations.
- First run a development pilot to verify finite predictions, training progress, modality alignment, runtime and VRAM. Expand to the prespecified matrix only after these checks.
- Core comparisons: QMMF-Net, HeMIS-style 2.5D and channel-concatenation U-Net 2.5D. Component ablations: no quality conditioning, no variance, no max branch, no consistency, shuffled quality. A matched-capacity fusion control is required before attributing gains to fusion alone.
- Train all compared models on matching cases, crops and schedules. Use full-volume reconstruction for evaluation, with a check that inference does not discard labelled anatomy. A shallow 3D-context wrapper cannot stand in for a fully trained volumetric nnU-Net/SegResNet comparator.
- Primary endpoint: case-level macro Dice across WT/TC/ET under complete modalities. Robustness endpoint: mean case-level macro Dice across all 15 nonempty modality subsets. Report per-region scores and the number of evaluable cases for empty-reference rules.
- Use a prespecified, case-ID-based development subset for expensive pilot robustness checks. Full-study robustness must identify its exact cohort and cannot silently describe a small subset as all cases.
- Patient/case-paired bootstrap confidence intervals; multiplicity adjustment for prespecified primary comparisons. Preserve failures, empty predictions and non-finite metric counts. Do not derive confirmatory significance from four overlapping training folds.
- Exact Shapley values explain model performance changes across modality subsets, not clinical causality. Corruption experiments use fixed types, severities and seeds. Any calibration uses development cases only and the actual selected model's normalization.
- Full conference claims additionally require adequate repeated training, strong current comparators and independent validation with documented cohort overlap checks. MSD overlaps historical BraTS cohorts, so another BraTS mirror cannot automatically be called external validation.

## Compute and release rules

Start with one private GPU pilot and measure throughput before allocating a larger matrix. Use available Kaggle accelerators and existing account quota; do not purchase compute. Save per-run checkpoints, histories and a status ledger so interrupted runs can resume. Budget exhaustion produces an incomplete status, never a completed-study label.

Separate Markdown and PDF documents will cover: notebook audit, datasets, experiment plan, notebook operation, actual results, and GitHub release. The repository will include configuration, source, tests for scientific invariants, generated notebooks, dependency versions and figure/table scripts. Exclude tokens, raw MRI data, caches and large checkpoints from Git. A remote GitHub destination has not yet been specified; prepare a reviewable local release package first.

## Source anchors

- [Official Kaggle CLI](https://github.com/Kaggle/kaggle-cli)
- [Medical Segmentation Decathlon paper](https://www.nature.com/articles/s41467-022-30695-9)
- [HeMIS paper](https://arxiv.org/abs/1607.05194)
- [MRI slice-level data leakage study](https://www.nature.com/articles/s41598-021-01681-w)

## Status at the initial draft

Plan written; source audit in progress. No new experimental result is claimed yet. Any protocol amendments must record their reason and whether held-out results had already been observed.

### Amendment 1 — observed compute allocation, before new training

The initially serialized Kaggle quota response appeared to report 21,600 GPU seconds (six hours), zero used/reserved, resetting 2026-09-12 00:00 UTC. **This interpretation was later corrected in Amendment 3 because the SDK's duration serialization drops whole days.** Classification will run its seven-variant, five-fold, three-seed matrix within a one-hour session budget. Segmentation starts with a corrected development pilot and saves reusable cache/checkpoints. A full segmentation matrix is contingent on measured remaining compute; incomplete or pilot results must not be promoted to full-study claims. No new held-out predictions had been observed when this amendment was written.

The classification no-multiscale control keeps the three branches and the parameter count, setting every dilation to one. This isolates receptive-field diversity from capacity. Similarity groups use a fixed pHash distance <= 4 and 32x32 intensity correlation >= 0.98, resulting in 209 groups from 228 unique images in the local audit.

### Amendment 2 — CPU preprocessing and development study, before corrected segmentation training

Run segmentation integrity checks and cache construction in a private Kaggle **CPU script job**, then attach its versioned output to notebook 02. This preserves GPU quota and does not add a third research notebook. The first classification attempt failed before training because an accelerator identifier with incorrect capitalization fell back to P100, which is incompatible with the installed CUDA 12.8 PyTorch wheel. Use the documented `NvidiaTeslaT4` identifier and record the failed attempt.

The finite-budget segmentation study is explicitly **development evidence**: fold 0, seed 42, nine variants (QMMF, HeMIS-style, U-Net 2.5D, no quality, no variance, no max, no consistency, shuffled quality, matched-capacity moment fusion). The pilot schedule is 8 epochs x 40 microbatches for three core variants; the extended development schedule is 24 epochs x 100 microbatches for all nine. Crop 192x192, batch 4, accumulation 2; validate every four epochs on four fixed inner cases and three fixed modality subsets. Evaluate selected checkpoints on all fold-0 outer cases and all 15 modality subsets on eight fixed outer cases. The eight-case robustness analysis is exploratory and is labeled with its exact cohort size. Keep the 96-case locked test closed. This does not replace the larger repeated-seed, multi-fold conference protocol above.

### Amendment 3 — corrected duration accounting, before corrected segmentation outcomes

At 2026-09-08 17:16 UTC, direct inspection of the typed quota response showed `timedelta(days=1, seconds=21600)`: **108,000 seconds (30 GPU hours)**. The installed SDK's `to_dict()` output omitted the day component and also misformatted fractional seconds. The helper now uses `total_seconds()` and has a regression check for the observed failure. Actual use at this check was 500.074 seconds, leaving 107,499.926 seconds (about 29 hours 52 minutes), with zero GPU time reserved. The earlier six-hour interpretation was wrong; the completed classification results are unaffected.

The already prepared three-model pilot remains unchanged. Its measured training, validation and inference costs will determine the expanded allocation. The nine-variant development matrix remains the initial comparison set. The additional allowance should prioritize adequate training progress, a larger inner validation cohort and repeated seeds for the primary comparisons before adding more model names. Freeze the revised schedule and cohort sizes in a further amendment before the expanded runs. Any locked-test use requires a frozen final protocol after development; no corrected segmentation outcome had been observed at this amendment.

### Amendment 4 — rescaled duplicate cases, before any segmentation held-out result

After preparation completed, independent metadata review found six identical annotation-array pairs, five crossing the original fold-0 roles. Two sampled pairs have identical targets, support and crop geometry, and normalized image correlation greater than 0.999999 in all four channels. The original raw-array hash test misses these intensity-rescaled copies. The 484-case-ID split is therefore not independence-safe. The first GPU pilot was cancelled (session 348299981); its log records one training epoch for QMMF and HeMIS and no validation or test outcome. Discard these training states for subsequent reported fits.

Run a separate CPU similarity audit on the reusable caches. It samples a fixed stride-4 grid in original coordinates and compares all case pairs using per-modality Pearson correlation and brain-support Dice. Before observing that audit's full results, fix the added high-similarity edge rule to correlation >=0.995 in every modality and support Dice >=0.995. Conservatively keep all original Hamming <=4 candidates together as well. Connected components define split groups; they are conservative image groups, not verified patient IDs.

Rebuild stratified development/inner/outer/locked partitions at group level, with every related case confined to one role. Protect the fresh locked cohort by excluding all groups represented in the cancelled pilot's 244-case training/normalization list from its eligible pool. If that leaves insufficient eligible groups, report the limitation instead of silently relaxing it. For evaluation of multiple cases per group, use group-aware aggregation and uncertainty, or a prespecified representative per group; state the chosen rule before training. Freeze revised hashes and cohorts, fix fingerprint CSV text loading, verify group isolation, and then run a fresh pilot. The earlier case-ID split is retained solely as failed-protocol provenance.

### Amendment 5 — frozen group protocol, before fresh pilot training

The corrected float64 CPU similarity audit completed with 484 case IDs in 262 conservative image groups (largest 36). There are 610 coarse candidate pairs, 114 pairs satisfying the strong four-channel correlation/support criterion, and no additional strong edges outside the coarse screen. The first audit used float32 sums that allowed correlations slightly above one; those values are invalid and are superseded by the float64 rerun. A regression compares long sparse vectors against an independent correlation calculation.

The repaired split hash is **b92f9c7c0fb4c116**. All 171 groups represented in the cancelled pilot's 244 training IDs are ineligible for the fresh protected test. Of the remaining 91 eligible groups, stratified allocation selects 51 groups (66 cases) for that test. This deliberately restricted eligibility may shift the test distribution and must be disclosed. It provides untouched image groups, not independently verified patients or an external cohort. The development pool contains 211 groups (418 cases). Fold 0 has 132 training groups/294 cases, 24 inner-validation groups/33 cases, and 55 outer-validation groups/91 cases.

Training draws a group uniformly, then a case uniformly within that group. Fit the robust quality scaler using one within-group median descriptor matrix per training group. The shuffled-quality control uses a fixed derangement of training groups, with every donor belonging to a different training group. Each model receives matching group draws for a given seed. Report the full-modality primary endpoint as nonempty-region macro Dice per case, mean within group, then equal mean across groups; retain the case-weighted mean as a labeled secondary statistic. Bootstrap the aligned group means, with identical group draws for paired model differences.

For the fresh pilot, a fixed label-blind SHA256 representative rule selects one case per group. Use four fixed inner-group representatives for checkpoint selection on the same three modality subsets, all 91 outer cases for full-modality evaluation, and eight fixed outer-group representatives for all 15 nonempty subsets. Those eight groups are a pilot robustness sample. The exact case lists and all four grouped folds are frozen in `segmentation/configs/grouped_v2`. Larger nested representative lists (12 inner and 16 outer groups) are recorded for an extended development stage; its actual schedule remains contingent on measured pilot cost and learning. The protected test stays closed. Fresh pilot: QMMF, HeMIS-style and no-quality control, 8 epochs x 40 microbatches, identical optimizer/crop/loss schedule apart from the named ablation, seed 42. All 138 current tests and a real-metadata context check pass before submission.

### Amendment 6 — lossless training I/O, during the fresh pilot

A local CPU profile on the previously exposed case BRATS_275 measured median whole-cache read time 0.0960 seconds versus 0.00748 seconds for an eight-slice compressed chunk. This is a single-case warm-cache I/O measurement, not an end-to-end training speedup. It motivates a supporting CPU staging job that splits the existing float16 image/uint8 targets into compressed axial chunks. It reads back and verifies every image/target voxel for every converted case; hashes and a completion ledger are required before attaching this format to an extended study.

The training reader preserves exact image/target values, boundary padding, group/case/slice RNG sequence, modality draws, augmentation and quality descriptors. Full-volume evaluation continues to read the original canonical caches. Tests compare all axial boundary/chunk crossings and full augmented training samples between the two readers. This optimization changes storage access only. The fresh pilot continues with its original embedded reader, source and schedule. The longer study's resolved configuration records the chosen format and inventory hash; no expanded model fit is launched by this cache conversion.

Before selecting the longer schedule, run a bounded two-T4 throughput script for QMMF and HeMIS. For each model, compare the two cache readers using 10 warmup and 100 timed microbatches, batch four, accumulation two, the actual training augmentation, and active full-image consistency at curriculum epoch 25 of 120. Reverse reader order between models to expose order effects. Use only fold-0 training groups, discard every benchmark weight, and produce no validation/test score. This measures a small training segment including data loading and worker startup; it is not a complete fit or an inference benchmark. The separate supporting script has a 900-second session cap.

Kaggle documents a 12-hour CPU/GPU session limit and 20 GB of saved output space. Keep study batches below that time limit with room for checkpoint/output finalization; use measured runtime to allocate the 30-hour account allowance. [Official notebook specifications](https://www.kaggle.com/docs/notebooks).

### Amendment 7 — repeated-seed main study and protected evaluation, before main fits

The corrected pilot completed all three fits and passed independent checks for group isolation, training-only normalization, selected checkpoints, all 15 modality subsets and exact Shapley efficiency. Full-modality group Dice was 0.2520 (QMMF), 0.2461 (HeMIS-style) and 0.2499 (no quality). Every paired full/robustness interval includes zero; quality conditioning has no established benefit here. ET performance and worst-subset scores remain poor after only 160 scheduled optimizer steps. These pilot outcomes have been observed; they justify adequate training, not selecting a favorable variant or claiming convergence.

The real two-T4 training benchmark measured 0.2534 seconds/microbatch for QMMF with chunks versus 1.0176 with the original reader; HeMIS measured 0.2840 versus 0.9966. Peak training memory was 1.35 GiB and 0.74 GiB respectively. These finite-loss segments use the actual training pipeline with active consistency, 10 warmup and 100 timed microbatches. Reader order is reversed between models; this is one bounded timing comparison, not a distribution of hardware performance. All benchmark weights are discarded.

Freeze the main development matrix at **nine variants x three training seeds (42, 43, 44): 27 fits** on grouped fold 0. Every fit starts from fresh weights, trains up to **120 epochs x 100 microbatches**, with batch four/accumulation two (48,000 central training windows and 6,000 scheduled optimizer-step opportunities at the maximum). Use the same AdamW, 3e-4 learning rate, 1e-3 weight decay, warmup/cosine schedule, augmentation, modality curriculum and loss coefficients, except the explicitly named ablation. Validate every **10 epochs on 12 fixed inner-group representatives**, using the unchanged three-subset composite rule. Patience is eight validation checks. Evaluate all 91 outer cases/55 groups with full modalities and all 15 subsets on the **16 fixed outer-group representatives**. No per-model threshold or hyperparameter search is added.

The variants remain QMMF, HeMIS-style, U-Net 2.5D, no quality, no variance, no max, no consistency, shuffled quality and matched-capacity equal-moment fusion. The matched control now constrains widths to multiples of eight, matching the alignment used by the other models, and enforces a 2% parameter tolerance. Derived widths [24,48,96,168] give 1,227,977 parameters versus QMMF's 1,221,465 (+0.53%). This selection uses parameter counts only. It does not match every operation or establish equal runtime; actual costs remain reported.

Measured pilot validation/inference and chunked training times give a conservative five-wave estimate of **7.44 hours per seed** on two T4s, or **22.33 GPU-session hours for all three seeds**. This is an estimate. Each seed gets a 29,000-second runner budget and 30,000-second Kaggle session cap. Use a separate private run identity for each seed so its completed outputs can be attached without relying on undocumented version-pinning syntax. At most two such Kaggle sessions run concurrently. Preserve incomplete checkpoints and finish any interrupted fit under its original configuration; do not remove poor runs or shorten selected models to improve a table. Reserve at least four hours of the current allowance for the protected evaluation; monitor actual quota and durations.

Also freeze `segmentation/configs/grouped_v2/locked_evaluation_protocol.json` before observing main-study outcomes. The protected comparison is **QMMF, HeMIS-style, no quality and U-Net 2.5D**, each with all three fixed training seeds, using their existing fold-0 checkpoints selected solely on inner validation. There is no retraining or normalization fit on protected cases. Full-modality evaluation uses all 66 cases/51 groups. All 15 modality subsets use one label-blind hash-selected representative from each of those 51 groups. Threshold is 0.5. Primary aggregation weights groups equally; 10,000 paired group-bootstrap draws retain all training seeds, with seed SD reported separately and Bonferroni intervals for the three controls per endpoint. The test remains closed while development runs are completed and independently verified. Its restricted eligibility and unverified patient identities limit the result to internal image-group generalization.

The main matrix, its protected comparisons and every planned failure remain reportable irrespective of which model ranks first. External validation, a fully trained contemporary volumetric comparator and clinical/novelty claims are outside the evidence produced by this bounded protocol and remain explicit follow-up requirements.

### Amendment 8 — effective auxiliary objectives, during the main fits

Recorded 2026-09-09 KST (2026-09-08 UTC), after the corrected pilot outcomes and main inner-validation progress had been observed, before any completed main outer-fold result or protected score had been reviewed. An architecture audit of the exact submitted source confirms that QMMF and all six same-family controls produce two training-only auxiliary prediction heads, weighted 0.5 and 0.25. HeMIS-style and U-Net 2.5D produce no auxiliary logits, despite the shared `deep_supervision` configuration flag. Real pilot histories independently confirm this difference. The audit uses untrained networks and synthetic inputs; it does not access MRI or fit another model.

The earlier references to identical loss schedules or shared coefficients require this qualification: the effective auxiliary objective differs across architecture families. QMMF-versus-HeMIS/U-Net comparisons evaluate the complete model and training setup together, so their differences cannot be attributed solely to fusion. Absolute total training losses across these families are not numerically comparable as one objective. The no-quality, shuffled-quality and matched-moment controls retain the same auxiliary heads as QMMF and support the narrower component comparisons; the no-consistency control explicitly removes its named term.

This amendment corrects the method description and interpretation. It changes no submitted source, training budget, checkpoint-selection rule, endpoint, protected cohort or comparison family. Preserve every planned run and disclose the effective head counts and parameter counts in `results/implementation_audit/`. An additional loss-matched cross-family study would be a separate experiment, not a claim established by the present results.

### Amendment 9 — fixed qualitative examples, before protected evaluation

Freeze `audit/qualitative_protocol.json` at 2026-09-08 20:09:59 UTC (2026-09-09 KST). Corrected pilot outcomes and main inner-validation progress have been observed; no main outer metric table or protected outcome has been reviewed. This adds illustrative panels to the already planned protected-evaluation job. It changes no model fit, selected checkpoint, threshold, quantitative endpoint, comparison family or cohort.

Use seed 42 for QMMF, HeMIS-style, no quality and U-Net 2.5D. Sort the 51 protected robustness representatives by SHA256 of `20260909|qualitative|<case_id>` and take the first three: **BRATS_143, BRATS_119 and BRATS_023**. Each belongs to a different protected group. Select the axial slice with the largest reference whole-tumor area, breaking ties by the lowest axial index; an empty whole-tumor reference uses the middle slice. The same case, image, slice and display window apply to every model. This rule favors visible tumor anatomy and does not make the three examples a representative performance sample.

Render normalized FLAIR with a fixed [-3, 3] display window, declared RAS orientation and nested-region overlays. Captions report independently checked whole-volume Dice. Repeat inference only for these three cases with each of the four seed-42 checkpoints; record this additional cost separately. Checks bind images to checkpoint hashes, the frozen case/seed rule, axial reference areas, common inputs/references, PNG checksums, geometry, rendered region counts and quantitative Dice. Tests use synthetic arrays only. Qualitative inference runs only after the existing 27-fit development gate passes.

Derived single-slice panels retain source attribution and CC BY-SA 4.0 terms, which the [MSD authors state for the dataset](https://www.nature.com/articles/s41467-022-30695-9). Full MRI volumes, caches and classification images remain excluded from the release. These illustrations supplement the full-cohort quantitative evidence and do not add a population-level claim.

### Amendment 10 — historical cohort exposure and sensitivity analysis

Frozen at 2026-09-09 07:03:11 UTC, after the first two main development seeds had been reviewed and before any reserved-cohort execution in the revised experiment. Source and saved-output inspection of the supplied notebook confirms manifest 2553ba21125ecb05 and ordered split a392d0e39514f976. Its recorded pilot trained fold 0 and validated the first 16 entries of that fold's inner list. Seven revised reserved groups include seven of those validation cases, or nine current cases when all case IDs in those conservative groups are counted. No reserved group overlaps the original active fold-0 training groups. The original source also prepares or loads normalizers for four folds, with 387 case IDs in their declared training-pool union; 40 current reserved cases in 36 groups have membership in that union. Only fold-0 model training is recorded.

The phrase untouched image groups in Amendment 5 was too broad and is retracted. Protected refers to exclusion from fitting and selection in the revised experiment. Revised fits start from fresh weights and use new training-only normalizers, so this historical record does not show leakage across their own partitions. It does limit claims of a historically untouched test set. No undocumented earlier use or patient independence can be established from the supplied artifacts.

Retain the complete 51-group/66-case quantitative cohort, all four models, all three seeds, existing checkpoints, threshold and all 15 modality subsets. Retain all three fixed qualitative cases; BRATS_143 was in the original validation subset. Add a separately labeled exploratory sensitivity using complete revised reserved groups only when none of their members belonged to the original development partition. This leaves 15 groups/18 full-modality cases and 15 of the existing robustness representatives, selected solely from metadata. The frozen file is `audit/historical_exposure_sensitivity_protocol.json`, SHA-256 d9de307c680fe7a50183cf3f6ab0752b465b9753be2f33222e2a9d17e355aaf1.

Use the existing full, mean-subset and worst-subset group tables after full independent verification. Retain all three training seeds within each of 10,000 paired group-bootstrap draws, use bootstrap seed 20260909, report seed SD separately, and report both unadjusted and Bonferroni intervals for the three controls per endpoint. The analysis adds no training or inference and changes no execution protocol. It was added after the initial plan and after some development outcomes were known; it is an exploratory sensitivity, not a replacement primary test. Report it regardless of its direction, alongside the full cohort. Its small selected sample does not establish external or clinical validation.

## Execution closure — 2026-09-09

All 105 classification fits, three corrected segmentation pilot fits and 27 main segmentation fits completed and passed independent artifact checks. The main matrix completed its gate at 07:44 UTC. The frozen reserved job then evaluated all 12 selected checkpoints; its quantitative and real qualitative artifacts passed verification and were first reviewed at 11:13 UTC. The historical sensitivity reproduced its frozen 15-group membership and completed on the verified tables. No model, seed, endpoint or fixed qualitative case was removed after observing results.

The full reserved cohort favors the no-quality control over QMMF on full, mean-subset and worst-subset Dice under the declared adjusted group comparisons. The smaller historical sensitivity leaves the full and mean-subset differences uncertain and still favors no quality on worst-subset Dice. These negative findings, empty-reference failures and all development ablations are retained in [the execution report](docs/RESULTS.md). This section records the outcome of the protocol; it does not change the preceding frozen amendments or turn the sensitivity into a confirmatory test.

Final reporting adds descriptive regional, all-subset, Shapley and resource tables from verified outputs, with no new training, inference or hypothesis tests. The code suite passes 186 tests. The release contains two research notebooks, readable source, frozen configurations, result tables, figures, provenance, and seven separate Markdown/PDF documents. At experiment closure, a GitHub destination and author/license details had not been supplied, so the package was prepared locally. Subsequent authorized GitHub publication is documented in the [release guide](docs/GITHUB_RELEASE.md). Broader external-cohort or modern-comparator claims remain outside this completed experiment.
