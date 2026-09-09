# Audit of the supplied MRI notebooks

Review date: 2026-09-09. Original notebooks are preserved in the parent folder. SHA-256 hashes are in `audit/original_hashes.json`; extracted source and text outputs are in `audit/`. Release QA found that the initial hash file was empty, so full-file hashes were recorded on 2026-09-09, with the actual timestamp retained. Every code cell and retained text output matches the snapshots extracted on 2026-09-08; a full-notebook hash from that initial extraction is unavailable.

## What the notebooks actually do

| Input | Task and data | Saved execution evidence |
|---|---|---|
| `brain-mri-01 (1) (1).ipynb` | Binary classification of JPEG/PNG MRI images; Navoneel yes/no dataset | 17 code cells, 1,357 source lines; completed five-fold comparison of four models |
| `buet-1.ipynb` | Multimodal 2.5D WT/TC/ET segmentation; MSD Task01 | 23 cells; embedded source archive; pilot stops during training, with no saved final evaluation |

These are separate studies. A yes/no image classifier cannot be evaluated with tumour-region Dice; a segmentation cohort containing tumour cases does not provide a healthy-control classification benchmark.

## Classification: findings and implications

| Finding | Evidence in the original | Implication and correction |
|---|---|---|
| Patient identity is unavailable | Discovery assigns the filename stem to `subject_id`; output reports 227 subjects from 228 images | Filename collisions are not repeated patients. New manifests use `image_id` and `group_id`, explicitly identifying image similarity only. |
| Large duplicate burden | 506 files, 278 byte duplicates removed, 228 remaining | Independently confirmed by byte and decoded-image hashes. New audit checks conflicting labels before deduplication and groups conservative near copies across splits. |
| One attention ablation only | MSCANet vs no-attention variant | Cannot distinguish attention, dilation diversity, augmentation and capacity effects. New study includes a 2x2 attention/multiscale factorial and an augmentation control. |
| Baseline naming can mislead | `ResNet2D18` uses base width 16 and GroupNorm, from scratch | Report as a narrow GroupNorm ResNet-style baseline, not standard pretrained ResNet-18. |
| Proposed method does not lead every metric | Saved mean accuracy: MSCANet 0.8158, ResNet baseline 0.8206 | The original evidence does not establish superiority. Retain negative and inconclusive comparisons. |
| Statistics treat overlapping CV fits as independent | Paired t/Wilcoxon tests on five folds; normal 1.96-based interval | Five folds share training observations. New output uses descriptive group-bootstrap intervals over aligned OOF predictions and does not present confirmatory clinical significance. |
| Figures select favorable outcomes | Best outer test fold chosen; Grad-CAM shown only for correctly classified examples | This biases presentation. New comparison figure includes all models, folds and seeds. Future example figures must use a fixed selection rule and show failures. |
| Limited reproducibility artifacts | Predictions retained in memory, but not exported with stable image IDs | New run exports sample-level probabilities, full nested split assignments, configurations, histories and checkpoints. |

The new local audit finds 228 unique images: 141 yes and 87 no, organized into 209 conservative similarity groups. Similarity uses pHash distance <= 4 plus Pearson correlation >= 0.98 on 32x32 grayscale images. Candidate pairs with distance <= 8 are exported for review. No exact cross-label conflict or unreadable image was found. This screening cannot establish patient independence.

### Saved classification results — provenance only

The following numbers come from the input notebook, not the corrected experiment. Values are fold mean +/- standard deviation.

| Original model | Accuracy | Macro F1 | AUROC |
|---|---|---|---|
| MSCANet2D | 0.8158 +/- 0.0806 | 0.8046 +/- 0.0836 | 0.8908 +/- 0.0671 |
| No attention | 0.7984 +/- 0.1122 | 0.7811 +/- 0.1143 | 0.8309 +/- 0.0767 |
| Plain CNN | 0.7894 +/- 0.0842 | 0.7723 +/- 0.0902 | 0.8493 +/- 0.0896 |
| Narrow GroupNorm ResNet | 0.8206 +/- 0.0533 | 0.8052 +/- 0.0581 | 0.8800 +/- 0.0497 |

The new preprocessing and similarity splits differ. A numerical change between old and new tables is not a controlled estimate of the effect of a single bug fix.

## Segmentation: findings and repairs

| Finding | Evidence in the original source | Action |
|---|---|---|
| Channel identity mismatch | Dataset metadata: FLAIR, T1w, t1gd, T2w. Configuration: T1, T1ce, T2, FLAIR. Cache never reorders channels. | Cache construction now resolves metadata aliases and applies permutation [1,2,3,0]. Missing-modality labels now match the input sequence. |
| Training/inference quality mismatch | Training `SliceDataset` receives a fitted normalizer; `predict_volume` builds `VolumeSliceDataset` without one | Fitted training normalization is attached to the model, saved in checkpoints and required for quality-conditioned inference. |
| Invalid predictions can appear as zero Dice | Raw quality includes values near 400,000; fp16 can overflow; thresholding NaNs yields false | Reject non-finite loss/logits/probabilities. The saved zero scores are consistent with a serious failure, but the original checkpoint is unavailable, so the exact historical cause is not proved. |
| EMA teacher lacks missing images | The student input is zeroed before training; trainer changes its availability mask to all ones for the teacher | Dataset now retains the same augmented full input separately. Teacher receives actual full modalities. |
| Shuffled-quality ablation does nothing | `shuffle_quality` is stored but never used in `__getitem__` | A deterministic derangement substitutes another training case's quality vector; no held-out donor is used. |
| Inference can crop away anatomy | Every slice is center-cropped to 192x192 before reconstruction | Inference now pads the complete brain field of view to a compatible size. Cache construction rejects loss of labeled voxels during the brain crop. |
| Validation is unnecessarily expensive | Every epoch evaluates five subsets and surface/lesion metrics over multiple full volumes | Selection computes Dice only, on fixed inner cases/subsets at a declared interval. Detailed surface metrics remain a separate evaluation capability. |
| Repeated decompression and full-volume conversion | Every random training window reloads an NPZ and converts the whole volume to float32 | Use a bounded cache and convert extracted windows only. Quality fitting reads the small quality member directly. CPU preprocessing produces reusable cache artifacts. |
| Half-precision masked entropy can be NaN | Epsilon 1e-8 underflows in fp16 for unavailable modalities | Softmax/moment reductions and entropy use float32 where needed. |
| Large loss reductions overflow in fp16 | A 4x3x192x192 confidence mask exceeds fp16's finite sum range; `logits.sum()*0` can also produce NaN | Consistency probabilities and deep-loss zero anchors use float32. A regression test checks a nonzero consistency loss and finite gradients at the actual crop size. |
| Accumulation and resume are incomplete | Floor division for optimizer steps; final microbatch group divided by full accumulation; history not restored | Correct partial accumulation and step count; restore saved history when resuming. |
| Headline Dice weighting differs from paired tests | Summary averages region means; tests average regions within each case | Compute region macro within case, then average cases within conservative image groups and weight groups equally; retain the case-weighted mean separately. |
| “Full” is not evidence of completion | `REPORTABLE` depends on the preset string, and the original queue does not add the declared ablations | New status checks completed artifacts against the exact planned variants. Development results stay labeled as development results. |
| Raw-array hashing misses rescaled copies | New real-data review finds six identical annotation pairs; sampled pairs have matching support/targets and image correlation >0.999999 in every normalized channel | First pilot cancelled. A complete float64 spatial audit yields 262 conservative groups; fresh split b92f9c7c0fb4c116 confines related cases to one role and excludes pilot-exposed groups from the protected test. |

The completed preparation audit confirms 484 structurally valid volumes, but this does not establish independence. Five repeated-annotation pairs cross the initial fold-0 roles, including two nominal locked-test/training overlaps. Evidence is retained in `results/segmentation_audit/`. All 610 coarse candidates are kept within groups, and all case pairs were screened for strong four-channel spatial similarity. The first float32 calculation slightly exceeded valid correlation bounds; it was replaced with float64 sums/products and verified against an independent long-vector regression. A separate serialization issue requires reading the 512-bit binary fingerprint column explicitly as text in pandas.

### Additional research limitations

The frozen main implementation has an effective-objective difference across model families: QMMF and its six component controls use two auxiliary deep-supervision heads (weights 0.5 and 0.25), while HeMIS-style and U-Net return no auxiliary logits. The shared configuration flag therefore does not imply identical losses. A synthetic architecture audit of the submitted source and the real pilot loss histories confirm this in `results/implementation_audit/`. Cross-family comparisons assess the complete model and training setup; quality/fusion attribution relies on same-family controls. Amendment 8 records this disclosure during main training, without changing the running experiments.

The original saved run reports 484 labeled cases, a 96-case locked test and 388 development cases. QMMF-Net is logged at zero validation score through epoch 20; HeMIS and U-Net show nonzero progress, and output ends while they are training. The saved notebook contains no final segmentation scores, calibration result, completed ablation table or locked-test result card. Its 116 passing unit tests did not cover the training/inference normalization contract or real channel semantics.

The original code selects the best scoring fold checkpoint per model, fits calibration on fold-0 inner cases regardless of the selected fold, and substitutes U-Net 2.5D when the named volumetric non-inferiority comparator is missing. These paths do not establish the stated confirmatory hypotheses. The revised finite-budget workflow reserves its evaluation cohort during fitting and selection and reports explicitly paired development cohorts.

A retrospective audit on 2026-09-09 confirms that the supplied notebook's logged manifest and ordered split hashes match the retained original partition. Its pilot used fold 0 and the first 16 inner cases for validation. Seven current reserved groups contain seven of those original cases, or nine current cases when all case IDs in those conservative groups are counted. None overlaps the original active fold-0 training groups. The source also prepares or loads normalization objects for four folds; their declared pools contain 387 cases, although only fold-0 model training is recorded. Revised fits use fresh weights and new training-only normalizers. The historical validation overlap qualifies the reserved cohort's interpretation and is not evidence that the revised fits crossed their own partitions.

Amendment 10 retracts the broad earlier description of all reserved groups as untouched. The completed report retains the full 51-group analysis and the separately frozen exploratory sensitivity on 15 groups/18 cases entirely outside the original development partition. Case-level history, source hashes and the original 16-case selection rule are in `results/segmentation_audit/original_notebook_exposure.json` and its CSV. Fixed illustrative case BRATS_143 was in the original validation subset; BRATS_119 was in its outer partition and BRATS_023 in its locked partition. These examples remain the prespecified illustrations and cannot establish population independence.

The handcrafted “quality” vector is computed after intensity normalization. Its SNR proxy sees zero background, and its coefficient of variation is sensitive to a near-zero mean. These values are image descriptors, not validated acquisition-quality measurements. Train-fitted scaling fixes numerical handling, but an acquisition-quality claim still requires appropriate controls and external quality validation. A quality ablation that fails to improve performance must be reported.

The repository contains shallow-context SegResNet/Swin wrappers and comments referring to a master plan not supplied with these files. Merely having these class names does not constitute a fair full-volume reproduction. The current study does not claim comparison against fully trained nnU-Net, ShaSpec or a modern state-of-the-art missing-modality method.

## Verification

Before modifications: 116 original segmentation tests passed locally. The current suite has 186 passing tests covering the scientific repairs, real-format CSV loading, group isolation, cancelled-pilot exposure exclusion, equal-group sampling/aggregation, cross-group quality donors, long-vector correlation accuracy, independent statistical verification, quota accounting, frozen protected-checkpoint evaluation, training-history validation, historical-cohort selection and qualitative-figure geometry/provenance. A context check also loads the actual frozen metadata. These code checks do not establish accuracy on real MRI. Kaggle runtime and actual results are documented separately in `RESULTS.md`.

## Research references

- [MRI slice-level leakage study](https://www.nature.com/articles/s41598-021-01681-w): supports the need to distinguish slice/image splits from patient splits.
- [MSD paper](https://www.nature.com/articles/s41467-022-30695-9): dataset/task context and historical BraTS overlap.
- [HeMIS](https://arxiv.org/abs/1607.05194): foundational missing-modality segmentation comparator.
- [nnU-Net](https://www.nature.com/articles/s41592-020-01008-z): strong self-configuring segmentation baseline to consider for a full study.
- [ShaSpec](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Multi-Modal_Learning_With_Missing_Modality_via_Shared-Specific_Feature_Modelling_CVPR_2023_paper.pdf) and [Missing as Masking, MICCAI 2024](https://papers.miccai.org/miccai-2024/520-Paper0067.html): relevant later methods. Published scores cannot be copied into a table with different data splits or training budgets.
