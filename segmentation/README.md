# Corrected segmentation implementation

This source was extracted from the supplied `buet-1.ipynb`, then reviewed and repaired. The original embedded README is preserved in `../audit/embedded_original_README.md`. Use the repository-level experiment plan, audit and notebook guide as the current protocol.

The active entry point is `scripts/conference_run.py`, packaged by `../scripts/build_notebooks.py`. It supports CPU preparation, a development pilot and an extended development ablation matrix. It keeps the locked test closed.

Legacy scripts and preset files are retained as provenance and reusable components. They are not evidence that every advertised baseline, calibration analysis or ablation has been run. In particular, the active workflow does not call the old `scripts/build_notebooks.py` or `scripts/run_worker.py`. The local `requirements.txt` is also historical; use the repository-root requirements and documented run environments for the corrected workflow.

Run tests from this directory with `PYTHONPATH=src python -m pytest tests -q`, or use the repository-wide command in the root README. Tests verify implementation properties using synthetic data; real results must come from the completed Kaggle execution artifacts.
