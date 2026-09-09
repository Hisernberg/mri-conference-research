import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location("combine_studies",
    Path(__file__).resolve().parents[1] / "scripts/combine_segmentation_studies.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def test_seed_variation_is_reported_without_multiplying_group_count():
    frames = {seed: pd.DataFrame({"qmmf": np.array([.3, .5, .7]) + shift,
                                  "control": np.array([.3, .5, .7]) + shift - .1},
                                 index=["g0", "g1", "g2"])
              for seed, shift in [(42, -.05), (43, 0), (44, .05)]}
    rows, contrasts, seed_rows = analysis.repeated_seed_summary(frames, "dice", draws=500)
    result = next(r for r in rows if r["model"] == "qmmf")
    assert result["n_groups"] == 3 and result["n_training_seeds"] == 3
    assert result["mean"] == pytest.approx(.5)
    assert result["seed_sd"] == pytest.approx(.05)
    assert len(seed_rows) == 6
    np.testing.assert_allclose([contrasts[0][k] for k in ["difference", "ci_low", "ci_high"]], .1)
    frames[44] = frames[44].rename(index={"g2": "other"})
    with pytest.raises(ValueError, match="alignment"):
        analysis.repeated_seed_summary(frames, "dice", draws=50)
