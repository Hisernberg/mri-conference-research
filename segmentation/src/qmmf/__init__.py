"""QMMF-Net: Quality-Conditioned Masked Moment Fusion for calibrated brain
tumour segmentation with arbitrary missing MRI modalities.

Reference implementation of the master research plan (protocol qmmf-v1.0).

Nothing in this package claims an experimental result. Every performance
number the plan mentions is a planning target or a decision gate; the code
produces measurements, and the notebooks record them.
"""

from .config import (
    DataConfig, EvalConfig, ExperimentConfig, LossConfig, MODALITIES,
    ModelConfig, OUTPUTS, PROTOCOL_VERSION, TrainingConfig,
)

__version__ = "1.0.0"
__protocol__ = PROTOCOL_VERSION

__all__ = [
    "ExperimentConfig", "DataConfig", "ModelConfig", "LossConfig",
    "TrainingConfig", "EvalConfig", "MODALITIES", "OUTPUTS",
    "PROTOCOL_VERSION", "__version__",
]
