# Conference writing guide

The repository supports writing a paper about the completed evaluation of quality conditioning for MRI segmentation with simulated missing modalities. Start with the [manuscript draft](MANUSCRIPT_DRAFT.md), [20-figure catalog](FIGURE_CATALOG.md), [editable tables](../paper/tables/) and [BibTeX references](../paper/references.bib). The classification notebook is a separate supporting benchmark. A venue and page limit have not been supplied, so this package uses a venue-neutral structure.

The main conclusion is a negative finding for the implemented quality descriptors and training schedule. In the complete reserved evaluation, the no-quality control has higher full, mean-subset and worst-subset Dice than QMMF under the declared paired comparisons. Do not rewrite the contribution as a successful quality module or a state-of-the-art claim.

## A compact paper structure

| Section | What to write | Direct evidence |
|---|---|---|
| Abstract | Research question, audit, executed scope, primary difference, limitations | Tables: reserved segmentation and contrasts |
| Introduction | Why incomplete-modality evaluation needs controlled evidence; specific quality-conditioning question | Related-work report and verified references |
| Data and audit | Source dataset, channel semantics, rescaled copies, grouping, historical use | Figures 01–02; dataset and notebook audit reports |
| Methods | Implemented fusion, same-family controls, shared training and differences in auxiliary objectives | Frozen source snapshots, resolved configurations, objective audit |
| Evaluation | Role boundaries, selected checkpoints, metric reductions, subset representatives, intervals | Frozen protocols, verification files |
| Main results | All nine development variants and all four reserved families | Figures 07–08 and 11–12; main/reserved tables |
| Robustness and failures | Every modality subset, regional denominators, empty-reference failures, fixed examples | Figures 13–14, 17 and 19 |
| Sensitivity | Complete reserved cohort plus the smaller historical sensitivity | Figure 18 and historical sensitivity table |
| Discussion | Unsupported quality benefit, comparison limits, practical failure patterns | Claim-evidence map |
| Reproducibility | Kaggle versions, input/source/output hashes, code and artifact access | Kaggle index and visualization execution receipt |

For a short paper, begin with six displayed figures: 02 (cohorts), 08 (main ablations), 11 (reserved endpoints), 14 (modality subsets), 18 (historical sensitivity) and 19 (real examples). Keep the failure result from Figure 17 in the main text or substitute it for Figure 19 when space is tight. Add a main-results table and the paired reserved comparisons. This is an editorial starting point, not a conference-specific limit. Preserve all 20 figures in the supplement or linked repository, including the negative and inconclusive findings. Gallery identifiers remain stable; renumber manuscript figures only after the final selection.

## Metric definitions to preserve

For full segmentation, compute a case's macro Dice over its nonempty reference regions, average cases within each conservative image group, weight groups equally and then summarize across the three fixed training seeds. WT, TC and ET regional tables use their own nonempty-reference denominators and cannot simply be averaged to reconstruct the case-first headline macro.

The main full-modality development endpoint uses 91 cases/55 groups. Its subset study uses 16 fixed representatives. Reserved full-modality evaluation uses 66 cases/51 groups; reserved subset evaluation uses one representative for each of the same 51 groups. The all-four-modality cell in the subset heatmap therefore uses 51 cases and differs slightly from the 66-case headline endpoint. QMMF is approximately 0.7705 in that heatmap cell and 0.7689 in the headline full-modality result.

Mean-subset Dice averages each representative's 15 nonempty subset scores before averaging groups. Worst-subset Dice takes each representative's minimum over those 15 subsets before averaging groups. Taking the minimum of the 15 column means defines a different quantity. Keep the declared order of reductions.

The reserved regional report uses 66 cases/51 groups for WT and TC, but 64 cases/49 groups for nonempty-reference ET. Both ET-empty cases receive false-positive ET from every model and seed. Their measured predicted volumes are shown separately in Figure 17.

## Statistical wording

Use “mean +/- sample standard deviation over seeds 42, 43 and 44” for seed variation. Use “paired group-bootstrap interval retaining the three fixed seeds” for comparison uncertainty. The 10,000 draws resample conservative image groups, not verified patients, independent slices or 153 independent seed/group observations.

The main segmentation comparisons adjust for eight QMMF-control contrasts separately per endpoint; the reserved comparisons adjust for three controls separately per endpoint. The smaller sensitivity uses the same three-control adjustment. This does not imply one simultaneous 95% guarantee across every endpoint, figure and exploratory analysis. Classifier follow-up intervals are descriptive and unadjusted. Do not infer a paired difference from overlapping marginal model intervals.

The 15-group historical sensitivity is exploratory. It was frozen before the reserved outcomes were examined, after some main development outcomes were available. Its cohort overlaps the complete 51-group report. Retain both and do not present the two estimates as independent replications.

## What each comparison can establish

The no-quality and shuffled-quality controls test descriptor contribution within the QMMF family. The matched-moment control tests a closer, capacity-matched alternative to the proposed fusion. The no-variance, no-max and no-consistency variants probe the implemented components. Preserve every declared variant, including controls with higher means.

The local HeMIS-style and U-Net implementations lack the two auxiliary heads used by QMMF and its same-family controls. Their comparisons combine architecture and effective objective differences. They do not isolate quality conditioning, and the local implementations must not be described as reproductions of every detail of their published counterparts.

Missingness is simulated after four-channel preprocessing and support/crop construction. The results do not validate an acquisition process that never obtained the missing sequence. Shapley attribution uses a declared zero empty-subset utility, not a measured empty-input prediction. Its values are descriptive under that game and are not causal clinical importance estimates.

## Files for writing

The [figure catalog](FIGURE_CATALOG.md) contains full captions, source links and interpretation boundaries. Each figure has PNG, PDF, SVG and plotted-data CSV files. Figures 01–18 and 20 have vector chart elements; Figure 19 preserves the original raster montage in PNG/PDF and uses a raster wrapper for SVG. Its [attribution](../results/conference_figures/ATTRIBUTION.md) must travel with the figure.

[Eight tables](../paper/tables/table_manifest.json) are generated directly from frozen CSV/JSON evidence and exported as CSV, Markdown and LaTeX fragments. The LaTeX fragments use `booktabs`; they are intended for insertion into the selected venue's template. The manuscript also has editable DOCX and review PDF copies. Formatting is a working draft, not an assertion of compliance with an unspecified venue.

Use the [claim-evidence CSV](../paper/claim_evidence.csv) when drafting results and discussion. It ties each proposed statement to a file, figure and limitation. Source papers are included in [references.bib](../paper/references.bib); the [related-work report](RELATED_WORK.md) records additional methods that were not reproduced here. Their published scores must not be imported into this study's comparison table.

## Information the authors must supply

Before submission, add the actual author list, affiliations, contact, contribution statement, funding and conflicts, institutional determination about secondary-data use, source-code rights/license, and the target venue's formatting and disclosure requirements. The supplied notebooks do not establish these facts. Do not invent an ethics approval number, exemption, patient consent statement, external cohort, DOI or conference acceptance.

The [CLAIM 2024 reporting guidance](https://pmc.ncbi.nlm.nih.gov/articles/PMC11304031/) is a useful reference for describing medical-imaging AI studies transparently. It does not certify this package, replace an institutional determination or establish venue compliance. This guide is an editorial aid; final scientific positioning and author declarations remain author decisions.
