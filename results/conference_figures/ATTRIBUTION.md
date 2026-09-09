# Attribution for derived qualitative figures

Data: Medical Segmentation Decathlon Task01_BrainTumour, a BraTS-derived cohort, supplied by the MSD/BraTS contributors. [Dataset mirror](https://www.kaggle.com/datasets/thisisrick25/medical-segmentation-decathlon-brain-tumour); [Antonelli et al., The Medical Segmentation Decathlon, 2022](https://www.nature.com/articles/s41467-022-30695-9).

The source authors identify the dataset license as [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). These derived panels retain that license. This notice applies to the data-derived figures; it does not assign a license to the research code.

Changes: canonical MRI preprocessing, normalized FLAIR display with a fixed [-3, 3] window, brain-support background masking, one axial slice selected by reference whole-tumor area, native RAS display orientation, and colored reference/prediction overlays. Three case IDs and training seed 42 were fixed before the revised reserved-cohort evaluation. Dice captions are whole-volume scores.
The supplied notebook's earlier validation subset included BRATS_143. Historical roles are documented in results/segmentation_audit/original_notebook_exposure.json; the fixed examples are current-model illustrations, not evidence of a historically untouched cohort.
