import numpy as np

from alphago.alphago import build_training_data
from alphago.evaluator import play
from alphago.estimator import create_trivial_estimator
from alphago.games import noughts_and_crosses as nac
from alphago.player import MCTSPlayer
from .games import mock_game


# TODO: mock lots of things in this file, especially players


def test_mcts_can_self_play_fake_game():

    player1 = MCTSPlayer(1, mock_game, mock_game.mock_estimator, 100, 0.5)
    player2 = MCTSPlayer(2, mock_game, mock_game.mock_estimator, 100, 0.5)
    players = {1: player1, 2: player2}

    states, action_probs = play(mock_game, players)

    assert states[0] == mock_game.INITIAL_STATE
    assert len(states) == 4
    assert len(action_probs) == 3


def test_mcts_can_self_play_noughts_and_crosses():
    estimator = create_trivial_estimator(nac.compute_next_states)
    player1 = MCTSPlayer(1, nac, estimator, 100, 0.5)
    player2 = MCTSPlayer(2, nac, estimator, 100, 0.5)
    players = {1: player1, 2: player2}

    game_states, action_probs = play(nac, players)

    assert len(action_probs) == len(game_states) - 1
    assert nac.is_terminal(game_states[-1])
    assert game_states[0] == nac.INITIAL_STATE


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
    [(1, {1: 0.5, 2: 0.5}, -7), (4, {3: 0.7}, 7), (3, {2: 0.3, 5: 0.7}, -7),
        (6, {1: 1.0}, 7)],
]


def test_build_training_data():
    mock_game.TERMINAL_STATE_VALUES = range(12)  # TODO: this is bad idea

    states = [0, 1, 3, 8]
    action_probs = [
        {0: 1 / 3, 1: 2 / 3},
        {0: 2 / 3, 1: 1 / 3},
        {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}]

    action_indices = {0: 0, 1: 1, 2: 2}

    training_data = build_training_data(states, action_probs,
                                        mock_game, action_indices)

    # The utility in terminal state 8 is {1: 1, 2: -1} in the mock game.
    expected = [(np.array(states[0]), np.array([1/3, 2/3, 0]), 1),
                (np.array(states[1]), np.array([2/3, 1/3, 0]), -1),
                (np.array(states[2]), np.array([1/3, 1/3, 1/3]), 1)]

    assert len(training_data) == len(expected)
    mock_game.TERMINAL_STATE_VALUES = (1,) * 12
    for comp, expec in zip(training_data, expected):
        assert (comp[0] == expec[0]).all()
        assert (comp[1] == expec[1]).all()
        assert comp[2] == expec[2]
