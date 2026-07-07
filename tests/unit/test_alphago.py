import numpy as np
from omegaconf import OmegaConf

from alphago.alphago import process_self_play_data, train_alphago
from alphago.config import Config, resolve_paths
from alphago.estimator import create_trivial_estimator
from alphago.evaluator import play
from alphago.games import NoughtsAndCrosses
from alphago.player import MCTSPlayer

from .games.mock_game import MockGame

# TODO: mock lots of things in this file, especially players


def test_mcts_can_self_play_fake_game():
    mock_game = MockGame()
    player1 = MCTSPlayer(mock_game, mock_game.mock_estimator, 100, 0.5)
    player2 = MCTSPlayer(mock_game, mock_game.mock_estimator, 100, 0.5)
    players = {1: player1, 2: player2}

    actions, game_states, utility = play(mock_game, players)

    assert len(actions) == 3
    assert game_states[0] == mock_game.initial_state
    assert len(game_states) == 4


def test_mcts_can_self_play_noughts_and_crosses():
    nac = NoughtsAndCrosses()
    estimator = create_trivial_estimator(nac)
    player1 = MCTSPlayer(nac, estimator, 100, 0.5)
    player2 = MCTSPlayer(nac, estimator, 100, 0.5)
    players = {1: player1, 2: player2}

    actions, game_states, utility = play(nac, players)

    assert len(actions) == len(game_states) - 1
    assert game_states[0] == nac.initial_state
    assert nac.is_terminal(game_states[-1])


TRAINING_DATA_STATES = [
    [1, 2, 3, 4],
    [1, 4, 3, 6, 7],
]

TRAINING_DATA_ACTION_PROBS = [
    [{1: 0.5, 2: 0.5}, {3: 0.7}, {2: 0.3, 5: 0.7}],
    [{1: 0.5, 2: 0.5}, {3: 0.7}, {2: 0.3, 5: 0.7}, {1: 1.0}],
]

TRAINING_DATA_ACTION_INDICES = [
    {1: 0, 2: 1, 3: 2, 4: 3, 5: 4},
    {1: 0, 2: 1, 3: 2, 4: 3, 5: 4},
]

TRAINING_DATA_EXPECTED = [
    [(1, {1: 0.5, 2: 0.5}, -4), (2, {3: 0.7}, 4), (3, {2: 0.3, 5: 0.7}, -4)],
    [
        (1, {1: 0.5, 2: 0.5}, -7),
        (4, {3: 0.7}, 7),
        (3, {2: 0.3, 5: 0.7}, -7),
        (6, {1: 1.0}, 7),
    ],
]


def test_process_self_play_data():
    mock_game = MockGame()
    mock_game.terminal_state_values = (1,) * 12

    states = [0, 1, 3, 8]
    actions = [0, 1, 0]
    action_probs = [
        {0: 1 / 3, 1: 2 / 3},
        {0: 2 / 3, 1: 1 / 3},
        {0: 1 / 3, 1: 1 / 3, 2: 1 / 3},
    ]

    action_indices = {0: 0, 1: 1, 2: 2}

    states_before = list(states)
    training_data = process_self_play_data(
        states, actions, action_probs, mock_game, action_indices
    )

    # process_self_play_data must not mutate the caller's states list.
    assert states == states_before

    # The utility in terminal state 8 is {1: 1, 2: -1} in the mock game.
    expected = [
        (np.array(states[0]), 0, np.array([1 / 3, 2 / 3, 0]), 1),
        (np.array(states[1]), 1, np.array([2 / 3, 1 / 3, 0]), -1),
        (np.array(states[2]), 0, np.array([1 / 3, 1 / 3, 1 / 3]), 1),
    ]

    assert len(training_data) == len(expected)
    mock_game.terminal_state_values = (1,) * 12
    for comp, expec in zip(training_data, expected, strict=False):
        assert (comp[0] == expec[0]).all()
        assert comp[1] == expec[1]
        assert (comp[2] == expec[2]).all()
        assert comp[3] == expec[3]


class FakeLogger:
    """Records every ``log_scalar`` call and whether ``close`` was called.

    Satisfies the ``MetricLogger`` seam so ``train_alphago`` can log losses and
    eval rates through it without importing a concrete backend.
    """

    def __init__(self):
        self.records = []
        self.closed = False

    def log_scalar(self, tag, value, step):
        self.records.append((tag, value, step))

    def close(self):
        self.closed = True


class FakeEstimator:
    """A lightweight estimator satisfying the contract ``train_alphago`` needs.

    ``create_estimate_fn`` returns the trivial uniform estimator so MCTS runs
    fast; ``train`` returns a fixed loss summary; ``save``/``restore`` are
    no-ops so no checkpoint files touch disk during the smoke.
    """

    def __init__(self, game):
        self._game = game

    def create_estimate_fn(self):
        return create_trivial_estimator(self._game)

    def train(
        self,
        training_data,
        batch_size,
        training_iters,
        mode="reinforcement",
        verbose=True,
    ):
        return {"total": 1.0, "value": 0.5, "policy": 0.5}

    def save(self, path):
        pass

    def restore(self, path):
        pass


def test_train_alphago_logs_on_alphago_step_axis(tmp_path):
    # A 1-step train_alphago smoke driven by a single cfg object and an injected
    # FakeLogger, proving that only plain primitives cross the seam and that
    # losses and eval rates are logged through the injected logger.
    nac = NoughtsAndCrosses()

    def create_estimator():
        return FakeEstimator(nac)

    cfg = OmegaConf.structured(Config)
    cfg.paths.experiment_dir = str(tmp_path)
    resolve_paths(cfg, "noughts_and_crosses")
    cfg.training.alphago_steps = 1
    cfg.training.self_play_iters = 30
    cfg.training.training_iters = 2
    cfg.training.evaluate_every = 1
    cfg.training.batch_size = 8
    cfg.training.num_evaluate_games = 2
    cfg.training.win_rate = 1.0
    cfg.mcts.mcts_iters = 2
    cfg.verbose = False

    fake = FakeLogger()
    train_alphago(nac, create_estimator, cfg, logger=fake)

    tags = [tag for tag, _, _ in fake.records]
    # A loss/* tag on the training step and eval/* tags on the eval step.
    assert any(tag.startswith("loss/") for tag in tags)
    assert any(tag.startswith("eval/") for tag in tags)
    # Everything logged on the integer alphago_step axis.
    assert all(isinstance(step, int) for _, _, step in fake.records)
    # The loop closed the injected logger.
    assert fake.closed
