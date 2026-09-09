"""The SDK's serialized durations omit days; budget against typed durations."""
from datetime import datetime, timedelta
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

SPEC = importlib.util.spec_from_file_location("mri_kaggle_api",
    Path(__file__).resolve().parents[1] / "scripts/kaggle_api.py")
api = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(api)


def test_quota_preserves_whole_days_and_subsecond_usage():
    quota = SimpleNamespace(time_used=timedelta(seconds=500, microseconds=74000),
        time_reserved=timedelta(hours=2), total_time_allowed=timedelta(days=1, hours=6),
        minimum_time_allowed=timedelta(days=1, hours=6))
    response = SimpleNamespace(quota_refresh_time=datetime(2026, 9, 12), gpu_quota=quota, tpu_quota=quota)
    result = api.quota_summary(response)["gpu"]
    assert result["total_time_allowed"] == 108000
    assert result["time_used"] == pytest.approx(500.074)
    assert result["remaining_unreserved"] == pytest.approx(100299.926)
