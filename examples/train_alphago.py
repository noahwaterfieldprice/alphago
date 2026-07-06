"""Config- and logger-driven Connect Four self-play training entry point."""

import time
from pathlib import Path

from omegaconf import OmegaConf

from alphago.alphago import train_alphago
from alphago.config import Config, load_config, resolve_paths
from alphago.device import enable_cpu_determinism, get_device
from alphago.estimator import ConnectFourNet
from alphago.games import ConnectFour
from alphago.metric_logger import WandbLogger


def main(cfg) -> None:
    """Run Connect Four self-play training from one resolved config.

    Builds the run directory and absolute paths, seeds RNGs on the CPU
    path, snapshots the resolved config to ``config.yaml``,
    injects a :class:`WandbLogger` built from a plain container,
    and calls :func:`train_alphago` with a primitive-only estimator closure so
    the config never crosses the estimator seam.

    Args:
        cfg: The resolved experiment configuration (built via
            ``load_config(Config)``).
    """
    game = ConnectFour()

    # Fill absolute experiment/checkpoint dirs when unset.
    resolve_paths(cfg, "connect_four")

    # Derive the wandb run name from the game + timestamp if the user left it
    # unset, so each run is identifiable in the backend.
    if cfg.wandb.name is None:
        stamp = time.strftime("%Y-%m-%d_%H:%M:%S")
        cfg.wandb.name = f"connect_four-{stamp}"

    # Seed RNGs only on the CPU path — enable_cpu_determinism is CPU-only and
    # must never run on MPS.
    if get_device(cfg.estimator.device).type == "cpu":
        enable_cpu_determinism(cfg.seed)

    # Create the run dir and snapshot the resolved config.
    experiment_dir = Path(cfg.paths.experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, experiment_dir / "config.yaml")

    # Build the metric logger from a PLAIN container — a raw DictConfig must not
    # be handed to wandb.
    logger = WandbLogger(cfg.wandb, OmegaConf.to_container(cfg, resolve=True))

    # Capture PLAIN primitives from cfg.estimator into the zero-arg factory;
    # cfg (or a subgroup) must never be passed into the constructor.
    learning_rate = cfg.estimator.learning_rate
    l2_weight = cfg.estimator.l2_weight
    value_weight = cfg.estimator.value_weight
    device = cfg.estimator.device
    action_indices = game.action_indices

    def create_estimator():
        return ConnectFourNet(
            learning_rate=learning_rate,
            l2_weight=l2_weight,
            value_weight=value_weight,
            device=device,
            action_indices=action_indices,
        )

    train_alphago(game, create_estimator, cfg, logger=logger)


if __name__ == "__main__":
    main(load_config(Config))
