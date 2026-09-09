# Dataset documentation

Prepared 2026-09-09. Counts are labeled by provenance: independently audited data, source metadata, or historical notebook output.

## Dataset A — binary 2D image classification

Source: [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection), owner `navoneel`, Kaggle dataset ID 165566. API metadata is retained locally under `data/classification/dataset-metadata.json`.

| Field | Verified value |
|---|---|
| Task | yes/no image classification; `yes` is the positive class |
| Raw image files | 506 |
| Unique bytes and decoded grayscale images | 228 |
| Duplicate files collapsed | 278 |
| Unique class counts | yes 141; no 87 |
| Conservative similarity groups | 209 |
| Exact cross-label conflicts | 0 |
| Unreadable files | 0 |
| Patient/session/scanner identities | Not supplied or verified |
| Source image geometry | Variable 2D image sizes; no verified voxel spacing |
| License field returned by Kaggle | `copyright-authors` |

The source tree includes duplicate copies under a wrapper folder. Do not count the duplicated directory tree as 506 independent examples. Hashing the filename is insufficient because copied images can have different names.

The audit checks SHA-256 of raw bytes and decoded grayscale pixel content including shape, and exports a candidate-pair table for approximate copies. Exact duplicates with conflicting labels would be quarantined. Approximate copies are kept as separate images within the same split group when pHash distance <= 4 and 32x32 intensity Pearson correlation >= 0.98. Grouping is conservative, deterministic and label-independent; it is not a patient-identification method.

Each experiment uses the same five outer group folds and an inner group validation split. The full manifest assigns every image a role for every fold. An image appears in exactly one outer test fold per training seed. No group crosses train, validation and test within a fold. The global split seed is 42; model seeds are 42, 43 and 44.

Preprocessing uses aspect-preserving grayscale letterboxing to 128x128, per-image foreground z-score normalization, and clipping to [-5,5]. Per-image normalization does not fit a cohort statistic. Augmentation occurs only on training batches. The no-augmentation control has identical evaluation preprocessing.

Audit manifest hash: `75d99ae5e0c343289d1148b892441f714fe97a549baa7c44f9dd3afa484e7415`.

Frozen split manifest hash: `008b31e71f5e86fec4d98c728fe7095038810e3eb509d56355f9c7eff86194b3`.

Do not describe results as patient-level sensitivity/specificity, diagnostic validation, clinical readiness or segmentation accuracy. The metrics describe the labeled images in this benchmark. Device, provenance and acquisition heterogeneity are insufficiently documented for subgroup analysis.

## Dataset B — MSD Task01 BrainTumour segmentation

Selected mirror: [Medical Segmentation Decathlon: Brain Tumour](https://www.kaggle.com/datasets/thisisrick25/medical-segmentation-decathlon-brain-tumour), owner `thisisrick25`, Kaggle dataset ID 8927583. Scientific source: [Medical Segmentation Decathlon](https://www.nature.com/articles/s41467-022-30695-9).

| Field | Source or verification rule |
|---|---|
| Task | Multimodal brain tumour segmentation |
| Labeled case IDs | 484 independently verified; these are not 484 verified independent patients |
| Conservative image-similarity groups | 262; largest group contains 36 case IDs |
| Reserved internal cohort | 51 groups, 66 case IDs; excludes cancelled-pilot training groups; earlier development exposure is documented below |
| Unlabeled challenge cases | No reference labels are assumed available; excluded from evaluation |
| Image format | 4D NIfTI, spatial volume plus four MRI channels |
| Source channel order | FLAIR, T1w, t1gd, T2w from `dataset.json` |
| Canonical model order | T1, T1ce, T2, FLAIR; permutation [1,2,3,0] |
| Source labels | 0 background, 1 edema, 2 non-enhancing tumour, 3 enhancing tumour |
| Model outputs | WT, TC, ET, overlapping binary regions |
| Mirror license field | `CC-BY-SA-4.0` in current Kaggle API metadata |
| Data origin limitation | Historical adult BraTS-derived cohort; another overlapping BraTS mirror is not independent validation |

Region derivation is explicit: WT = {1,2,3}; TC = {2,3}; ET = {3}. These integers differ from common BraTS legacy labels, where edema and core identities can differ and enhancing tumour may be label 4. The code validates the declared scheme before generating targets.

The CPU preparation job verifies image/label geometry, affines, orientation, finite values, modality count, label vocabulary, decoded-array duplicates and perceptual duplicate candidates. It writes a source fingerprint, complete manifest, channel-order record and split hash. Exact duplicate arrays fail the integrity gate. Approximate candidates require inspection and cannot prove patient identity. NIfTI filenames serve as dataset case IDs; any repeated-visit identity beyond supplied case IDs remains an external validation requirement.

**Independence audit failure:** the completed new audit finds 478 distinct annotation-array hashes, with six repeated pairs. Five pairs cross the initial fold-0 train/validation/test roles. Direct inspection of two pairs confirms identical targets and brain support, and correlation above 0.999999 for each normalized MRI channel. Rescaled intensities let copied images pass raw-array hashing. Thus the original 484-case case-ID split cannot support a patient-independent evaluation.

The completed float64 audit compares all case pairs on a stride-4 grid in original coordinates. All 610 coarse Hamming <=4 edges are conservatively grouped; 114 pairs also have correlation >=0.995 in all four channels and support Dice >=0.995. No strong pair adds an edge outside the coarse screen. Connected components yield 262 image groups, largest 36. The conservative rule can merge images that are not true copies; these are neither an estimated patient count nor guaranteed exhaustive patient identification. The earlier float32 correlation calculation exceeded one slightly because of long-sum error and is superseded by the verified float64 calculation.

The original development/test partition yielded 96 nominal locked-test cases and 388 development cases. Its hash, `a392d0e39514f976`, is unsuitable because image copies cross those boundaries. It remains solely as audit provenance. The replacement **b92f9c7c0fb4c116** stratifies complete groups by median WT volume and any ET presence. All 171 groups represented in the cancelled pilot's 244 training IDs are excluded from protected-test eligibility. From 91 eligible groups, the frozen split selects 51 groups/66 cases for the protected test; 211 groups/418 cases form development. This restricted eligibility may shift the protected test's distribution and does not create external validation.

The cancelled pilot produced no held-out score. Fresh fold 0 contains 132 training groups/294 cases, 24 inner groups/33 cases, and 55 outer groups/91 cases. The new pilot selects four inner-group representatives and eight outer-group representatives by a fixed label-blind hash rule; it evaluates all 91 outer cases with full modalities. Frozen lists and all four folds are in `segmentation/configs/grouped_v2`. Related case IDs never cross a role within a fold. Sampling is uniform over training groups, then uniform within the selected group; primary metrics average cases within groups, then groups equally.

**Historical exposure qualification:** the supplied notebook logs the same original manifest and ordered split hash. Its recorded pilot trained fold 0 and validated the first 16 inner cases. Seven of those case IDs belong to seven groups in the revised reserved cohort; those conservative groups contain nine current reserved case IDs in total. The source also prepares or loads normalizers for all four folds, whose declared training pools cover 387 case IDs. Only fold-0 model training is recorded.

| Recorded historical role or declared pool | Direct overlap among 66 reserved cases | Reserved groups with overlap | Reserved cases in those groups |
|---|---|---|---|
| Active fold-0 model training | 0 | 0 | 0 |
| 16-case validation subset | 7 | 7 | 9 |
| Across-fold normalization pools | 40 | 36 | 48 |

These rows overlap and are not additive. Revised fits use fresh weights and normalizers fitted on their own training groups; historical source use does not demonstrate leakage within those revised fits. It does prevent claiming that all reserved groups were historically untouched. Amendment 10 defines protected as reserved from fitting and selection in the revised experiment. The full 51-group report is retained, with an additional exploratory sensitivity analysis on 15 complete groups/18 cases whose members all belonged to the original locked partition. Its 15 robustness representatives and all models/seeds were frozen using metadata before revised reserved-cohort outcomes. The small selected subset does not provide external or patient-level validation. Evidence is in `results/segmentation_audit/original_notebook_exposure.json` and its case-level CSV.

Preprocessing crops to the union of nonzero modality support with a margin, verifies that annotated voxels are preserved, normalizes each modality per case, and stores version-2 compressed float16 cache entries with the verified channel order. Training extracts five adjacent slices around a central target slice. Inference covers every slice and pads the full brain field of view before reconstructing the original geometry.

Missingness is simulated by hiding channels from complete four-modality cases after shared preprocessing. The brain crop is derived from the complete input's nonzero support. These experiments measure robustness under that benchmark convention; they do not evaluate prospectively incomplete acquisitions with preprocessing derived only from available scans. The later chunk-cache conversion is lossless relative to the existing float16 canonical images and uint8 targets, not relative to the original unnormalized MRI intensities.

A completed CPU conversion stores the same training values in compressed eight-slice chunks for selective decompression. All 484 image/target arrays were compared voxel-for-voxel after writing. The inventory records both original and converted archive hashes. This storage optimization does not change the dataset, split, labels or augmentation; full-volume inference still uses the canonical original cache.

Quality normalization is fitted using one median descriptor matrix per training group, saved with the training case/group IDs and embedded in the selected checkpoint. The same transform is required at inference. The descriptor vector is not a validated scanner-quality or signal-to-noise measurement.

Descriptors are computed once per preprocessed complete volume and stored alongside the cached images. Training augmentation does not recompute these volume-level descriptors. Availability masks exclude absent modality tokens in the fusion calculation. The teacher receives the full augmented image and the corresponding stored descriptors, subject to the named quality ablation.

## Acquisition and redistribution

Download the datasets through Kaggle using the user's own authorized account. Full MRI volumes, NIfTI files, derived caches and classification images are excluded from the GitHub package. Link to the original dataset cards and preserve the returned license metadata and any applicable original attribution requirements. The classification metadata does not grant a general open-data redistribution license; the repository must not relabel it as MIT, CC0 or public domain.

The completed protected-evaluation figures include small, annotated single-slice panels derived from MSD Task01. The [MSD paper](https://www.nature.com/articles/s41467-022-30695-9) identifies the dataset license as CC BY-SA 4.0. These derived panels retain that license and credit the MSD/BraTS contributors, source paper and Kaggle mirror. Their attribution file describes normalization, slice selection, display orientation and overlays. This figure-specific notice does not assign a license to the research code. All 12 prediction panels passed checks of checkpoint identity, fixed case/slice selection, common input/reference, PNG hashes, orientation, mask counts and whole-volume Dice captions. BRATS_143's historical validation role remains disclosed.

The Kaggle metadata endpoint used here did not return an explicit dataset version number. Content hashes and the attached notebook source/output version provide the immediate reproducibility record. Do not invent a version number. Store the exact dataset version from Kaggle's attachment metadata when available.

The project does not have prospective consent, site-level provenance or external clinical validation information beyond the supplied public dataset records. This documentation concerns research benchmarking and does not establish clinical utility.
