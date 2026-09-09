from qmmf.config import ExperimentConfig
from qmmf.models import ABLATIONS, build_model, parameter_matched_widths


def test_aligned_fusion_control_matches_actual_model_capacity():
    cfg = ExperimentConfig()
    matched = parameter_matched_widths(cfg, "equal_mean_var", channel_multiple=8, tolerance=.02)
    assert matched["within_tolerance"] and all(w % 8 == 0 for w in matched["widths"])
    actual = ABLATIONS["A16"].apply(cfg).merged({"net.widths": matched["widths"]})
    assert sum(p.numel() for p in build_model(actual).parameters()) == matched["parameters"]
    assert abs(matched["ratio"] - 1) < .02
