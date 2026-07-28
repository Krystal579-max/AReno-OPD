"""Unit tests for OPD algorithm registration and configuration."""
from __future__ import annotations

import pytest


class TestOPDAlgorithmRegistration:
    """Test that the OPD algorithm is properly registered."""

    def test_opd_algorithm_registered(self):
        """The ``opd`` algorithm should be discoverable via the registry."""
        from areno.api.algorithms import get_algorithm, list_algorithms

        algorithms = list_algorithms(include_experimental=False)
        assert "opd" in algorithms, "OPD algorithm should be registered in built-in algorithms"

        spec = get_algorithm("opd")
        assert spec.name == "opd"
        assert spec.requires_rollout is True, "OPD is an on-policy algorithm and requires rollout"
        assert spec.experimental is False, "OPD is a built-in algorithm"

    def test_opd_trainer_resolves(self):
        """The OPD trainer class should resolve without error."""
        from areno.api.algorithms import get_algorithm

        spec = get_algorithm("opd")
        trainer_cls = spec.resolve_trainer_cls()
        assert trainer_cls.__name__ == "OPDTrainer", f"Expected OPDTrainer, got {trainer_cls.__name__}"

    def test_opd_loss_fn_factory(self):
        """The OPD loss function factory should bind hyperparameters."""
        from areno.api.algorithms import get_algorithm
        from areno.api.trainer_config import OPDTrainerConfig

        config = OPDTrainerConfig(
            algo="opd",
            ckpt="/path/to/ckpt",
            dataset_path="/path/to/data",
            opd_kl_coef=0.5,
            opd_temperature=2.0,
        )
        spec = get_algorithm("opd")
        loss_fn = spec.make_loss_fn(config)
        # The factory should have bound kl_coef=0.5 and temperature=2.0.
        import inspect
        sig = inspect.signature(loss_fn)
        bound = sig.bind({}, None)
        bound.apply_defaults()
        # Check that the partial function has the correct keyword arguments.
        assert loss_fn.keywords.get("kl_coef") == 0.5, f"Expected kl_coef=0.5, got {loss_fn.keywords.get('kl_coef')}"
        assert loss_fn.keywords.get("temperature") == 2.0, (
            f"Expected temperature=2.0, got {loss_fn.keywords.get('temperature')}"
        )


class TestOPDConfig:
    """Test the OPD trainer configuration dataclass."""

    def test_default_config(self):
        """OPDTrainerConfig should have sensible defaults."""
        from areno.api.trainer_config import OPDTrainerConfig

        config = OPDTrainerConfig(
            algo="opd",
            ckpt="/path/to/ckpt",
            dataset_path="/path/to/data",
        )
        assert config.ref_ckpt is None, "ref_ckpt should default to None"
        assert config.opd_kl_coef == 1.0, "opd_kl_coef should default to 1.0"
        assert config.opd_temperature == 1.0, "opd_temperature should default to 1.0"
        assert config.n_samples == 8, "n_samples should default to 8 (inherited from RolloutTrainerConfig)"
        assert config.requires_rollout  # via the algorithm spec

    def test_custom_config(self):
        """OPDTrainerConfig should accept custom hyperparameters."""
        from areno.api.trainer_config import OPDTrainerConfig

        config = OPDTrainerConfig(
            algo="opd",
            ckpt="/path/to/student",
            dataset_path="/path/to/data",
            ref_ckpt="/path/to/teacher",
            opd_kl_coef=0.1,
            opd_temperature=0.5,
            n_samples=4,
            batch_size=16,
        )
        assert config.ref_ckpt == "/path/to/teacher"
        assert config.opd_kl_coef == 0.1
        assert config.opd_temperature == 0.5
        assert config.n_samples == 4
        assert config.batch_size == 16

    def test_opd_config_inherits_rollout(self):
        """OPDTrainerConfig should inherit rollout fields from RolloutTrainerConfig."""
        from areno.api.trainer_config import OPDTrainerConfig

        config = OPDTrainerConfig(
            algo="opd",
            ckpt="/path/to/ckpt",
            dataset_path="/path/to/data",
            temperature=0.7,
            top_p=0.9,
        )
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.greedy is False

    def test_resolved_max_running_prompts(self):
        """resolved_max_running_prompts should default to batch_size * n_samples."""
        from areno.api.trainer_config import OPDTrainerConfig

        config = OPDTrainerConfig(
            algo="opd",
            ckpt="/path/to/ckpt",
            dataset_path="/path/to/data",
            batch_size=16,
            n_samples=4,
        )
        assert config.resolved_max_running_prompts() == 64

    def test_opd_areno_config(self):
        """OPDTrainerConfig should produce a valid backend config."""
        from areno.api.trainer_config import OPDTrainerConfig

        config = OPDTrainerConfig(
            algo="opd",
            ckpt="/path/to/ckpt",
            dataset_path="/path/to/data",
        )
        areno_cfg = config.areno_config()
        assert areno_cfg is not None
        assert hasattr(areno_cfg, "tp_size")
        assert hasattr(areno_cfg, "max_running_prompts")


class TestOPDLossExport:
    """Test that the OPD loss function is properly exported."""

    def test_opd_loss_in_loss_fns_init(self):
        """The opd_loss_fn should be importable from areno.api.loss_fns."""
        from areno.api.loss_fns import opd_loss_fn

        assert callable(opd_loss_fn)

    def test_opd_loss_in_algorithms_import(self):
        """The opd_loss_fn should be imported in algorithms.py."""
        from areno.api.algorithms import get_algorithm

        spec = get_algorithm("opd")
        assert spec.default_loss_fn is not None
        assert callable(spec.default_loss_fn)


class TestOPDTrainerExport:
    """Test that the OPD trainer is properly exported."""

    def test_opd_trainer_in_trainers_init(self):
        """The OPDTrainer should be importable from areno.api.trainers."""
        from areno.api.trainers import OPDTrainer

        assert OPDTrainer.__name__ == "OPDTrainer"

    def test_opd_trainer_in_all(self):
        """The OPDTrainer should be listed in __all__."""
        from areno.api.trainers import __all__ as trainers_all

        assert "OPDTrainer" in trainers_all