# Related work and conference positioning

Checked 2026-09-09 using primary papers, proceedings and author repositories. This is a focused comparison for the supplied project, not an exhaustive novelty search. Published scores from different cohorts are not inserted into this project's result tables.

## Relevant methods

| Method | Established idea | Implication for this study |
|---|---|---|
| [HeMIS, 2016](https://arxiv.org/abs/1607.05194) | Aggregate modality-specific latent representations to handle missing inputs | Moment-based fusion is established prior work. The local comparator is explicitly a HeMIS-style 2.5D implementation. |
| [ShaSpec, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Multi-Modal_Learning_With_Missing_Modality_via_Shared-Specific_Feature_Modelling_CVPR_2023_paper.pdf) | Separate shared and modality-specific representations for incomplete multimodal learning | Shared encoding and modality-aware representations alone are insufficient novelty claims. ShaSpec has not been reproduced in the current results. |
| [M3AE, AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/25253) | Mask modalities and patches for pretraining, then use self-distillation during supervised fine-tuning | Pretraining data, additional optimization and missing-input training recipes must be accounted for in a fair comparator. |
| [M3FeCon, MICCAI 2024](https://papers.miccai.org/miccai-2024/520-Paper0067.html) | Reconstruct missing modality features using a masking formulation | Feature completion is a relevant alternative to direct available-modality fusion. A comparison needs compatible training missingness. |
| [DC-Seg, MICCAI 2025](https://papers.miccai.org/miccai-2025/0213-Paper0653.html) | Disentangle anatomical and modality-specific representations using contrastive objectives | Robustness and representation-learning claims need comparison against later methods, not only basic U-Nets. |
| [IM-Fuse, MICCAI 2025](https://papers.miccai.org/miccai-2025/0437-Paper0747.html) | Study Mamba fusion and re-evaluate missing-modality methods on a larger BraTS2023 cohort | Rankings can depend on dataset scale. Its author benchmark is a useful reference for a broader reproduction effort. |
| [SimMLM, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Li_SimMLM_A_Simple_Framework_for_Multi-modal_Learning_with_Missing_Modality_ICCV_2025_paper.html) | Dynamic modality-expert weighting and a ranking objective comparing more versus fewer inputs | Dynamic gating and reliability under missing inputs are already studied. The proposed quality-conditioned feature moments require a narrower, empirically supported contribution. |
| [D3Seg, 2026 preprint](https://arxiv.org/abs/2605.22249) | Combine modality-dependency graphs, latent diffusion imputation and decision refinement | Recent alternatives continue to expand; a claim of state of the art requires an updated, directly comparable benchmark. This source is a preprint, not assumed conference acceptance. |

The IM-Fuse paper links an author repository that currently redirects to [MiMoSe](https://github.com/AImageLab-zip/MiMoSe). SimMLM provides its [official implementation](https://github.com/LezJ/SimMLM). Repository availability does not establish a completed reproduction here; no score from these repositories is claimed as this project's experiment.

## Defensible questions for the supplied project

The strongest immediate question is whether quality-conditioned moments provide a reproducible benefit after correcting modality semantics, inference normalization, teacher inputs and split contamination. The no-quality, shuffled-quality, no-variance, no-max and matched-parameter controls directly address that question. A result that does not support the quality contribution must lead to a narrower claim.

The local HeMIS-style and U-Net comparators have no auxiliary deep-supervision heads, while QMMF and its same-family controls use two. Their cross-family contrasts therefore evaluate architecture and effective auxiliary objective together. The matched-moment, no-quality and shuffled-quality controls preserve QMMF's auxiliary heads and provide the closer tests of the fusion and descriptor claims; a cross-family improvement alone cannot isolate either contribution.

An additional question is whether learned gating weights agree with measured changes in segmentation utility across the 15 modality subsets. SimMLM also discusses gate-weight interpretation; therefore gating visualizations alone are not a new contribution. Exact subset-based Shapley values can support an exploratory comparison in this project, with the declared empty-set utility and explicit uncertainty. They do not establish causal clinical importance. [SimMLM full text and appendix](https://arxiv.org/html/2507.19264v2).

The rescaled-copy discovery makes dataset handling a prerequisite. More training on the original case-ID split would not resolve that flaw. The repaired benchmark must report both case and conservative group counts, preserve the failed split's provenance, and prevent related volumes from crossing roles.

## Requirements for a broader comparator

A modern comparator should use an identified author implementation and revision, documented data access and code terms, verified channel/label mapping, the same protected groups, and a declared optimization and selection budget. Record external pretraining separately. A shallow-context wrapper must not be labeled a fully trained volumetric nnU-Net or another published 3D model.

The completed Kaggle work has not established superiority over the methods above, independent external-cohort generalization, architectural novelty or conference acceptance. It provides all 27 planned development fits and 12 reserved checkpoint evaluations. Quality conditioning does not improve the main study; the no-quality control has better reserved full, mean-subset and worst-subset scores under the declared adjusted group comparisons. The small historical sensitivity leaves the full comparison uncertain and retains a no-quality worst-subset advantage. A defensible report must retain these negative findings. Further claims require an author-led assessment of the closest methods and evidence beyond this internal benchmark.
