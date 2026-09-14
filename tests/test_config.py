"""Tests for EdgeEye configuration loading and validation."""

import importlib
import os
import sys

import pytest


def reload_config(env_overrides: dict) -> object:
    """Reload src.config with a clean environment snapshot.

    Temporarily sets/unsets env vars, forces a module reload, then
    restores the original environment.  Returns the freshly-loaded module.
    """
    original = {}
    for key, value in env_overrides.items():
        original[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    # Force a fresh import so _load_confidence_threshold() re-runs
    if "src.config" in sys.modules:
        del sys.modules["src.config"]

    try:
        import src.config as config
        return config
    finally:
        # Restore original environment
        for key, original_value in original.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value

        # Clean up the reloaded module so it doesn't bleed into other tests
        if "src.config" in sys.modules:
            del sys.modules["src.config"]


class TestConfidenceThresholdDefault:
    """CONFIDENCE_THRESHOLD should default to 0.5 when env var is not set."""

    def test_default_value(self):
        config = reload_config({"CONFIDENCE_THRESHOLD": None})
        assert config.CONFIDENCE_THRESHOLD == 0.5

    def test_default_is_float(self):
        config = reload_config({"CONFIDENCE_THRESHOLD": None})
        assert isinstance(config.CONFIDENCE_THRESHOLD, float)


class TestConfidenceThresholdOverride:
    """CONFIDENCE_THRESHOLD should use the value from the environment."""

    def test_override_to_zero_point_three(self):
        config = reload_config({"CONFIDENCE_THRESHOLD": "0.3"})
        assert config.CONFIDENCE_THRESHOLD == pytest.approx(0.3)

    def test_override_to_zero(self):
        config = reload_config({"CONFIDENCE_THRESHOLD": "0.0"})
        assert config.CONFIDENCE_THRESHOLD == 0.0

    def test_override_to_one(self):
        config = reload_config({"CONFIDENCE_THRESHOLD": "1.0"})
        assert config.CONFIDENCE_THRESHOLD == 1.0

    def test_override_to_high_precision(self):
        config = reload_config({"CONFIDENCE_THRESHOLD": "0.75"})
        assert config.CONFIDENCE_THRESHOLD == pytest.approx(0.75)


class TestConfidenceThresholdValidation:
    """Invalid values should raise ValueError at import time."""

    def test_above_one_raises(self):
        with pytest.raises(ValueError, match="0.0 and 1.0"):
            reload_config({"CONFIDENCE_THRESHOLD": "1.1"})

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="0.0 and 1.0"):
            reload_config({"CONFIDENCE_THRESHOLD": "-0.1"})

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError, match="must be a number"):
            reload_config({"CONFIDENCE_THRESHOLD": "high"})

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="must be a number"):
            reload_config({"CONFIDENCE_THRESHOLD": ""})
