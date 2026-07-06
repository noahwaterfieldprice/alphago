"""Unit tests for the OmegaConf config schema and loader.

Covers the structured schema (builds with the pinned defaults) and the loader
(struct-mode fail-loud on unknown keys / type mismatches, MISSING fail-loud, and
absolute run-dir derivation). One behavior per test, docstring per test, and
``pytest.raises`` for the fail-loud contracts -- mirroring device_test.py.
"""

import os
from dataclasses import dataclass

import pytest
from omegaconf import MISSING, OmegaConf
from omegaconf.errors import ConfigKeyError, MissingMandatoryValue, ValidationError

from alphago.config import Config, load_config, resolve_paths


def test_config_builds_with_defaults():
    """OmegaConf.structured(Config) exposes the pinned defaults across groups."""
    cfg = OmegaConf.structured(Config)
    assert cfg.training.batch_size == 32
    assert cfg.mcts.c_puct == 1.0
    assert cfg.estimator.learning_rate == 1e-3
    assert cfg.seed == 0
    assert cfg.verbose is True


def test_dotlist_override_applies():
    """A CLI dotlist overrides values with type coercion (int) preserved."""
    cfg = load_config(Config, ["training.batch_size=64", "estimator.device=mps"])
    assert cfg.training.batch_size == 64
    assert isinstance(cfg.training.batch_size, int)
    assert cfg.estimator.device == "mps"


def test_unknown_key_raises():
    """An unknown key is rejected at merge time (struct mode)."""
    with pytest.raises(ConfigKeyError):
        load_config(Config, ["training.bogus=1"])


def test_type_mismatch_raises():
    """A non-coercible value for a typed field fails loud (validation)."""
    with pytest.raises(ValidationError):
        load_config(Config, ["training.batch_size=notanint"])


def test_missing_field_fails_loud():
    """Accessing an unset MISSING-defaulted field raises (fail-loud).

    Locks the MISSING contract that Plan 05's SL script relies on for
    ``paths.training_data`` without adding a MISSING field to the real Config.
    """

    @dataclass
    class NeedsInput:
        training_data: str = MISSING

    cfg = OmegaConf.structured(NeedsInput)
    with pytest.raises(MissingMandatoryValue):
        _ = cfg.training_data


def test_resolve_paths_absolute():
    """resolve_paths derives absolute experiment_dir + nested checkpoint_dir."""
    cfg = OmegaConf.structured(Config)
    resolve_paths(cfg, "connect_four")
    assert os.path.isabs(cfg.paths.experiment_dir)
    assert os.path.isabs(cfg.paths.checkpoint_dir)
    assert cfg.paths.checkpoint_dir.startswith(cfg.paths.experiment_dir)
