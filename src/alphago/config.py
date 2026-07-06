"""OmegaConf-backed experiment configuration schema and loader."""

import time
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


@dataclass
class TrainingConfig:
    """Self-play training-loop hyperparameters.

    Attributes:
        self_play_iters: Self-play games generated per AlphaGo step.
        training_iters: Optimizer steps per network-optimization phase.
        alphago_steps: Number of self-play/train/evaluate iterations.
        evaluate_every: Run champion evaluation every N AlphaGo steps.
        batch_size: Mini-batch size for network optimization.
        replay_length: Maximum replay-buffer length (most recent samples).
        num_evaluate_games: Games played in each champion evaluation.
        win_rate: Win-rate threshold a challenger must beat to be promoted.
        restore_dir: A PREVIOUS run's checkpoint dir to restore from.
        restore_step: The checkpoint step within ``restore_dir`` to restore.
    """

    self_play_iters: int = 20
    training_iters: int = 20000
    alphago_steps: int = 1000
    evaluate_every: int = 10
    batch_size: int = 32
    replay_length: int = 20000
    num_evaluate_games: int = 50
    win_rate: float = 0.6
    restore_dir: str | None = None
    restore_step: int | None = None


@dataclass
class MctsConfig:
    """MCTS search hyperparameters.

    Attributes:
        mcts_iters: Simulations run per move.
        c_puct: PUCT exploration constant.
    """

    mcts_iters: int = 500
    c_puct: float = 1.0


@dataclass
class EstimatorConfig:
    """Neural-network estimator hyperparameters.

    Attributes:
        learning_rate: Optimizer learning rate.
        l2_weight: L2 regularization weight.
        value_weight: Relative weight of the value loss term.
        device: ``None`` -> CPU per the ``device.py`` contract.
    """

    learning_rate: float = 1e-3
    l2_weight: float = 1e-4
    value_weight: float = 1.0
    device: str | None = None


@dataclass
class PathsConfig:
    """Experiment filesystem paths.

    Both fields are derived by :func:`resolve_paths` when left ``None``.

    Attributes:
        experiment_dir: Absolute run directory for this experiment.
        checkpoint_dir: Absolute checkpoint directory nested under the run dir.
    """

    experiment_dir: str | None = None
    checkpoint_dir: str | None = None


@dataclass
class WandbConfig:
    """Weights & Biases logging configuration.

    Attributes:
        project: W&B project name.
        name: Run name; auto-derived at the entry point from game + timestamp.
        mode: One of ``online`` | ``offline`` | ``disabled``.
    """

    project: str = "alphago"
    name: str | None = None
    mode: str = "online"


@dataclass
class Config:
    """Top-level experiment configuration.

    Attributes:
        training: Self-play training-loop hyperparameters.
        mcts: MCTS search hyperparameters.
        estimator: Neural-network estimator hyperparameters.
        paths: Experiment filesystem paths.
        wandb: Weights & Biases logging configuration.
        seed: Global RNG seed.
        verbose: Whether to emit progress output.
    """

    training: TrainingConfig = field(default_factory=TrainingConfig)
    mcts: MctsConfig = field(default_factory=MctsConfig)
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    seed: int = 0
    verbose: bool = True


def load_config(schema, argv: list[str] | None = None) -> DictConfig:
    """Merge CLI dotlist overrides over a structured schema in struct mode.

    Builds ``OmegaConf.structured(schema)`` -- which is already in struct mode,
    so unknown keys and type mismatches raise on merge -- then merges
    ``OmegaConf.from_cli(argv)`` over it. The helper is generic over ``schema``
    so play/SL scripts can reuse it with their own dataclasses.

    Args:
        schema: A dataclass (or instance) describing the config shape.
        argv: A dotlist of ``key=value`` overrides. ``None`` reads
            ``sys.argv[1:]``.

    Returns:
        The merged configuration.

    Raises:
        Exception: An unknown key or a type mismatch in ``argv`` (struct mode
            and type validation fail loud at startup).
    """
    base = OmegaConf.structured(schema)
    cli = OmegaConf.from_cli(argv)
    return OmegaConf.merge(base, cli)


def resolve_paths(cfg, game_name: str) -> None:
    """Derive absolute experiment and checkpoint dirs in place when unset.

    Centralizes the ``experiments/<game>-<timestamp>/`` run-dir construction as
    an absolute path, building the run-dir string exactly once. Fields already
    set by the user are left untouched.

    Args:
        cfg: A config with a ``paths`` group (mutated in place).
        game_name: The game name embedded in the run-dir path.
    """
    if cfg.paths.experiment_dir is None:
        stamp = time.strftime("experiment-%Y-%m-%d_%H:%M:%S")
        run_dir = Path(f"experiments/{game_name}-{stamp}")
        cfg.paths.experiment_dir = str(run_dir.resolve())
    if cfg.paths.checkpoint_dir is None:
        cfg.paths.checkpoint_dir = str(
            (Path(cfg.paths.experiment_dir) / "checkpoints").resolve()
        )
