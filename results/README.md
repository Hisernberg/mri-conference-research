# Results and evidence index

Start with the [complete report](../docs/RESULTS.md) or [results PDF](../docs/pdf/RESULTS.pdf). The [completion record](../audit/completion_record.json) records the verified matrices and artifact hashes. Classification and segmentation use different datasets and endpoints; their scores are not combined.

| Folder | Contents |
|---|---|
| [conference_figures](conference_figures/) | 20 Kaggle PNG/PDF/SVG exports, plotted CSVs, executed notebook, complete captions and PDF atlas |
| [classification](classification/) | 105 fits, OOF probabilities, seed metrics, paired intervals, error analysis and figures |
| [classification_audit](classification_audit/) | Image manifests, duplicate/similarity groups and frozen partitions |
| [segmentation_audit](segmentation_audit/) | Structural and similarity audits, historical exposure, failed split provenance and repaired group metadata |
| [segmentation_grouped_pilot](segmentation_grouped_pilot/) | Three corrected short-training fits, regional results, histories and feasibility evidence |
| [segmentation_study_s42](segmentation_study_s42/), [s43](segmentation_study_s43/), [s44](segmentation_study_s44/) | All nine main variants for each fixed seed, case/subset metrics and independent verification |
| [segmentation_main](segmentation_main/) | Complete 27-fit comparison, adjusted ablation intervals, regional summaries, learning curves and resource tables |
| [segmentation_protected](segmentation_protected/) | All 12 selected checkpoint evaluations on 66 cases/51 reserved groups and all 15 modality subsets |
| [segmentation_protected/reporting](segmentation_protected/reporting/) | Descriptive regional, subset, Shapley and execution-cost tables |
| [segmentation_protected/qualitative](segmentation_protected/qualitative/) | Three prespecified real MRI examples, verified captions and CC BY-SA attribution |
| [segmentation_historical_sensitivity](segmentation_historical_sensitivity/) | Frozen exploratory 15-group/18-case sensitivity, retaining all four models and three seeds |
| [optimization](optimization/) | Lossless cache checks, Kaggle throughput measurements and budget estimates |
| [implementation_audit](implementation_audit/) | Effective auxiliary-objective differences and parameter-count evidence |

The reserved comparisons favor removing quality conditioning. The smaller historical sensitivity leaves full and mean-subset differences uncertain and favors no quality on worst-subset Dice. Every declared main variant remains reportable. Empty-reference ET failures and the historical validation overlap are retained in the report.

`verification.json` files describe the checks performed for their respective artifacts. Group-bootstrap intervals retain the fixed training seeds; seed SD is reported separately. The groups are conservative image-similarity units and are not verified patients. Synthetic fixtures in the test suite establish code behavior, not MRI accuracy.
