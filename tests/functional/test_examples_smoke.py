"""Struct-mode-safe runnability smoke tests for every touched example.

Each converted ``examples/`` script is loaded via ``importlib`` (the ``examples/``
directory is not a package) and its ``main(...)`` is called with tiny overrides so
the full entry-point path runs in seconds under pytest. These tests double as the
permanent struct-mode regression guard: a schema or struct-mode break fails these
CI-visible tests rather than a user's real run.

Every config-driven script is forced offline with ``wandb.mode=disabled`` so no
wandb run is created and nothing leaves the machine. The interactive play
scripts are driven through their move loop with a mocked ``input()``.
"""

import importlib.util
import itertools
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


def load_example(script_name: str):
    """Load an ``examples/`` script as a module via its file path.

    The ``examples/`` directory is not an importable package, so each script is
    loaded through ``importlib.util.spec_from_file_location``. The module's
    ``__name__`` is its stem (never ``"__main__"``), so the script's
    ``if __name__ == "__main__"`` block does not execute on import.

    Args:
        script_name: The script filename, e.g. ``"train_alphago.py"``.

    Returns:
        The loaded module object, exposing its ``main`` and config schema.
    """
    path = EXAMPLES_DIR / script_name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_alphago_smoke(tmp_path, monkeypatch):
    """Connect Four training runs the full self-play -> optimise -> evaluate ->
    checkpoint path and asserts the checkpoint lands inside checkpoint_dir.

    ``self_play_iters=15`` clears the hardcoded 100-row ``continue`` guard (a
    worst-case 7-move Connect Four game yields ~7 rows, so 15 games give ~105
    rows), so the loop reaches ``optimise_estimator``, ``logger.log_scalar``,
    ``evaluate_model``, and ``checkpoint_model``. With ``alphago_steps=1`` the
    loop runs step 0 and ``0 % evaluate_every == 0``, so a checkpoint is written
    to ``<checkpoint_dir>/0.pt`` — the permanent regression guard for the
    checkpoint-path join.
    """
    monkeypatch.chdir(tmp_path)
    module = load_example("train_alphago.py")

    cfg = module.load_config(
        module.Config,
        [
            "training.alphago_steps=1",
            "training.self_play_iters=15",
            "training.training_iters=2",
            "training.evaluate_every=1",
            "training.num_evaluate_games=1",
            "training.replay_length=100",
            "mcts.mcts_iters=2",
            "wandb.mode=disabled",
        ],
    )

    module.main(cfg)

    # main() resolves cfg.paths in place, so checkpoint_dir is an absolute path
    # under tmp_path. Before the path-join fix the file would land at a sibling
    # (<experiment_dir>/checkpoints0.pt), so this assertion fails loud.
    checkpoint_file = Path(cfg.paths.checkpoint_dir) / "0.pt"
    assert checkpoint_file.is_file()


def test_train_connect_four_sl_smoke(tmp_path, monkeypatch):
    """Supervised Connect Four training runs one step over a tiny data fixture."""
    monkeypatch.chdir(tmp_path)

    # A tiny, valid solver-data fixture (``<moves> <opt_action> <value>``); enough
    # lines that the 2% dev split rounds up to at least one sample.
    lines = []
    for i in range(80):
        col = (i % 7) + 1
        opt = ((i + 3) % 7) + 1
        value = (i % 3) - 1
        lines.append(f"{col} {opt} {value}")
    data_file = tmp_path / "sl_data.txt"
    data_file.write_text("\n".join(lines) + "\n")

    module = load_example("train_connect_four_sl.py")

    cfg = module.load_config(
        module.SupervisedConfig,
        [
            f"paths.training_data={data_file}",
            "num_steps=1",
            "batch_size=4",
            "mcts_iters=2",
        ],
    )

    module.main(cfg)


def _write_sl_fixture(path: Path, num_rows: int) -> None:
    """Write a tiny, valid solver-data fixture (``<moves> <opt_action> <value>``)."""
    lines = []
    for i in range(num_rows):
        col = (i % 7) + 1
        opt = ((i + 3) % 7) + 1
        value = (i % 3) - 1
        lines.append(f"{col} {opt} {value}")
    path.write_text("\n".join(lines) + "\n")


def test_train_connect_four_sl_tiny_dev_split_does_not_collapse(tmp_path, monkeypatch):
    """A <50-row dataset must not collapse the dev split to zero rows.

    With 40 rows, ``int(0.02 * 40) == 0``; before the ``num_dev = max(1, ...)``
    guard this yields an empty dev split and ``estimator.loss([], ...)`` /
    ``compute_accuracy`` crash. The ``max(1, ...)`` guard keeps at least one
    held-out row, so ``main()`` runs to completion.
    """
    monkeypatch.chdir(tmp_path)

    data_file = tmp_path / "sl_data.txt"
    _write_sl_fixture(data_file, num_rows=40)

    module = load_example("train_connect_four_sl.py")

    cfg = module.load_config(
        module.SupervisedConfig,
        [
            f"paths.training_data={data_file}",
            "num_steps=1",
            "batch_size=4",
            "mcts_iters=2",
        ],
    )

    module.main(cfg)


def test_train_connect_four_sl_evaluate_without_step_fails_loud(tmp_path, monkeypatch):
    """An evaluate path without ``evaluate_step`` fails loud before training."""
    monkeypatch.chdir(tmp_path)

    data_file = tmp_path / "sl_data.txt"
    _write_sl_fixture(data_file, num_rows=40)

    module = load_example("train_connect_four_sl.py")

    cfg = module.load_config(
        module.SupervisedConfig,
        [
            f"paths.training_data={data_file}",
            "evaluate_checkpoint_path=/tmp/does-not-need-to-exist",
        ],
    )
    # evaluate_step defaults to None, so this is the misconfigured case.
    assert cfg.evaluate_step is None

    with pytest.raises(RuntimeError):
        module.main(cfg)


def test_comparator_smoke(tmp_path, monkeypatch):
    """The NAC comparator runs a real self-play training run then compares vs optimal.

    NAC self-play with the real net runs end-to-end. ``self_play_iters``
    is raised to clear the hardcoded 100-row ``continue`` guard (a NAC game yields ~7
    rows, so ~20 games give ~140 rows), so the loop reaches ``optimise_estimator``,
    ``evaluate_model``, and ``checkpoint_model``. With ``alphago_steps=1`` and
    ``0 % evaluate_every == 0`` a checkpoint is written to ``<checkpoint_dir>/0.pt`` —
    the permanent regression guard that the comparator actually trained (not just that
    the script exited).
    """
    monkeypatch.chdir(tmp_path)
    module = load_example("alphago_noughts_and_crosses_comparator.py")

    cfg = module.load_config(
        module.Config,
        [
            "training.alphago_steps=1",
            "training.self_play_iters=20",
            "training.training_iters=2",
            "training.evaluate_every=1",
            "training.num_evaluate_games=1",
            "training.replay_length=100",
            "mcts.mcts_iters=2",
            "wandb.mode=disabled",
        ],
    )

    module.main(cfg)

    # main() resolves cfg.paths in place, so checkpoint_dir is an absolute path
    # under tmp_path. A checkpoint at step 0 only lands if the 100-row training
    # guard was cleared and optimise -> evaluate -> checkpoint ran, so this
    # asserts the comparator actually trained end-to-end.
    checkpoint_file = Path(cfg.paths.checkpoint_dir) / "0.pt"
    assert checkpoint_file.is_file()


def test_play_tournament_smoke(tmp_path, monkeypatch):
    """The Connect Four gauntlet runs one round and saves its plot into tmp."""
    monkeypatch.chdir(tmp_path)
    module = load_example("play_tournament.py")

    module.main(
        num_rounds=1,
        mcts_iters=2,
        output_path=str(tmp_path / "results.png"),
    )


def test_self_play_noughts_and_crosses_smoke(tmp_path, monkeypatch):
    """The 3x6 NAC self-play demo runs a single game each way."""
    monkeypatch.chdir(tmp_path)
    module = load_example("self_play_noughts_and_crosses.py")

    module.main(num_games=1, max_iters=2)


def test_play_connect_four_smoke(tmp_path, monkeypatch):
    """Interactive Connect Four runs its move loop with mocked human input."""
    monkeypatch.chdir(tmp_path)
    module = load_example("play_connect_four.py")

    # Cycle through the 7 columns so the human always eventually enters a legal,
    # non-full column and the game reaches a terminal state.
    moves = itertools.cycle(["1", "2", "3", "4", "5", "6", "7"])
    monkeypatch.setattr("builtins.input", lambda *_: next(moves))

    cfg = module.load_config(
        module.PlayConnectFourConfig,
        ["player=1", "mcts_iters=2"],
    )

    module.main(cfg)


def test_play_noughts_and_crosses_smoke(tmp_path, monkeypatch):
    """Interactive noughts and crosses runs its move loop with mocked input."""
    monkeypatch.chdir(tmp_path)
    module = load_example("play_noughts_and_crosses.py")

    # Cycle through the 9 cells so the human always eventually enters a legal,
    # empty cell and the 3x3 game reaches a terminal state.
    moves = itertools.cycle(["0", "1", "2", "3", "4", "5", "6", "7", "8"])
    monkeypatch.setattr("builtins.input", lambda *_: next(moves))

    cfg = module.load_config(
        module.PlayNoughtsAndCrossesConfig,
        ["player=1", "mcts_iters=2"],
    )

    module.main(cfg)
