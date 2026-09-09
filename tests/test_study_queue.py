"""A running GPU reservation must not be charged twice when scheduling seed 44."""
import importlib.util
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_remaining_study import submission_reserve


@pytest.mark.parametrize("active,reserved,expected", [
    (6000, 0, 50700), (6000, 2000, 48700), (6000, 9000, 44700)])
def test_reservation_only_adds_uncovered_active_time(active, reserved, expected):
    assert submission_reserve(active, reserved) == expected
