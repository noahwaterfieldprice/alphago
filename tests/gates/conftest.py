"""Shared fixture for the NAC behavioral gates.

A single seeded ~2.5-minute NAC 3x3 self-play training run produces BOTH the
recorded training logs AND the trained champion estimator, so all three gate
tests (E2E three-signal, vs-Random, vs-Optimal) reuse one run rather than
retraining per test (one run, three signals).

The run is fully seeded via ``cfg.seed`` (``enable_cpu_determinism`` before any
estimator is built) and fully offline via ``wandb.mode=disabled``. The
tuned training/mcts config is pinned as explicit module constants below; a
post-tuning failure is a discovered pipeline bug, never a threshold to relax.
"""

from dataclasses import dataclass

import pytest
from recording_logger import RecordingLogger

from alphago.alphago import compute_checkpoint_name, train_alphago
from alphago.config import Config, load_config
from alphago.device import enable_cpu_determinism
from alphago.estimator import NACNetEstimator
from alphago.games.noughts_and_crosses import NoughtsAndCrosses

# --- Pinned tuned gate config (empirically tuned within the ~15-min CPU
# ceiling). This exact config produces a champion that clears every
# downstream bar.
SEED = 0
MCTS_ITERS = 50
WIN_RATE = 0.55
LEARNING_RATE = 0.02

TRAIN_OVERRIDES = [
    "training.alphago_steps=20",
    "training.self_play_iters=80",
    "training.training_iters=800",
    "training.evaluate_every=1",
    "training.num_evaluate_games=20",
    f"training.win_rate={WIN_RATE}",
    "training.replay_length=20000",
    "training.batch_size=32",
    f"mcts.mcts_iters={MCTS_ITERS}",
    f"estimator.learning_rate={LEARNING_RATE}",
    f"seed={SEED}",
    "verbose=false",
    "wandb.mode=disabled",
]


@dataclass
class NacAgentRun:
    """The artifacts of one seeded NAC 3x3 training run, shared by every gate.

    Attributes:
        records: The ``RecordingLogger`` records (loss/eval scalar histories).
        cfg: The resolved config the run used (exposes ``seed``, ``mcts.c_puct``,
            ``training.win_rate``).
        game: The ``NoughtsAndCrosses`` instance the run trained on.
        champion: The trained champion estimator (the last self-play estimator
            promoted when ``eval/success_rate`` crossed ``win_rate``).
        mcts_iters: The pinned MCTS simulation count the gate agent must use.
    """

    records: dict[str, list[tuple[int, float]]]
    cfg: Config
    game: NoughtsAndCrosses
    champion: NACNetEstimator
    mcts_iters: int


@pytest.fixture(scope="session")
def nac_agent_run(tmp_path_factory) -> NacAgentRun:
    """Train ONE seeded, tuned NAC 3x3 run and expose its logs + champion.

    Seeds all RNGs via ``enable_cpu_determinism(cfg.seed)`` BEFORE building any
    estimator (so weight init is reproducible), runs ``train_alphago`` once with
    an injected :class:`RecordingLogger`, then restores the champion from the
    checkpoint at the last step where ``eval/success_rate`` crossed ``win_rate``
    (that is exactly the step ``train_alphago`` promotes the self-play estimator
    from). The single run backs all three gate tests.
    """
    ckpt_dir = tmp_path_factory.mktemp("nac_gate")
    cfg = load_config(
        Config,
        TRAIN_OVERRIDES
        + [
            f"paths.experiment_dir={ckpt_dir}",
            f"paths.checkpoint_dir={ckpt_dir}",
        ],
    )

    enable_cpu_determinism(cfg.seed)
    game = NoughtsAndCrosses()

    def create_estimator() -> NACNetEstimator:
        return NACNetEstimator(
            learning_rate=cfg.estimator.learning_rate,
            l2_weight=cfg.estimator.l2_weight,
            value_weight=cfg.estimator.value_weight,
            action_indices=game.action_indices,
            device=cfg.estimator.device,
        )

    recorder = RecordingLogger()
    train_alphago(game, create_estimator, cfg, logger=recorder)

    # The champion is the self-play estimator promoted at the last step whose
    # success_rate crossed win_rate (train_alphago restores it from that step's
    # checkpoint). Fall back to the final eval step if no swap fired, so the E2E
    # gate — not a fixture crash — surfaces a champion-never-fired failure.
    success = recorder.records.get("eval/success_rate", [])
    promoted_steps = [step for step, rate in success if rate > cfg.training.win_rate]
    champion_step = promoted_steps[-1] if promoted_steps else success[-1][0]

    champion = create_estimator()
    champion.restore(compute_checkpoint_name(champion_step, cfg.paths.checkpoint_dir))

    return NacAgentRun(
        records=recorder.records,
        cfg=cfg,
        game=game,
        champion=champion,
        mcts_iters=MCTS_ITERS,
    )
