import pytest

import alphago.games.noughts_and_crosses as nac
from alphago import mcts_tree
from alphago.evaluator import create_trivial_evaluator
from ..unit.games.mock_game import MockGame, mock_evaluator


def test_can_create_mcts_node():
    game_state = None
    node = mcts_tree.MCTSNode(game_state, player=1)
    assert node is not None


def test_can_create_mcts_tree():
    game_state = None
    tree = mcts_tree.MCTSNode(game_state, player=1)
    assert tree is not None


def test_can_run_mcts_on_fake_game():
    """ This test shows that we can run MCTS using a 'next_states'
    function and 'evaluator' function.
    """
    mock_game = MockGame()
    root = mcts_tree.MCTSNode(0, player=1)
    action_probs = mcts_tree.mcts(
        root, mock_evaluator, mock_game.compute_next_states, mock_game.utility,
        mock_game.which_player, mock_game.is_terminal, 100, 1.0)

    assert action_probs is not None


def mock_evaluator_action0(state):
    prior_probs = {action: prob for action, prob
                   in zip(range(3), [1, 0, 0])}
    value = 0
    return prior_probs, value


def mock_evaluator_action1(state):
    prior_probs = {action: prob for action, prob
                   in zip(range(3), [0, 1, 0])}
    value = 0
    return prior_probs, value


@pytest.mark.parametrize("evaluator, expected", [
    (mock_evaluator_action0, [0, 1, 3, 7]),
    (mock_evaluator_action1, [0, 2, 6, 17]),
])
def test_mcts_can_play_fake_game(evaluator, expected):
    root = mcts_tree.MCTSNode(0, player=1)
    node = root
    nodes = [node]
    mock_game = MockGame()

    while not node.is_terminal:
        action_probs = mcts_tree.mcts(
            root, evaluator, mock_game.compute_next_states, mock_game.utility,
            mock_game.which_player, mock_game.is_terminal, 100, 1)

        action = max(action_probs, key=action_probs.get)
        node = node.children[action]
        nodes.append(node)
    assert [node.game_state for node in nodes] == expected


def test_mcts_can_self_play_fake_game():
    mock_game = MockGame()
    states, action_probs = mcts_tree.self_play(
        mock_game.compute_next_states, mock_evaluator, mock_game.initial_state,
        mock_game.utility, mock_game.which_player, mock_game.is_terminal, 100, 1)

    assert states[0] == mock_game.initial_state
    assert len(states) == 4
    assert len(action_probs) == 3


def test_mcts_can_self_play_noughts_and_crosses():
    evaluator = create_trivial_evaluator(nac.compute_next_states)

    game_states_, action_probs_ = mcts_tree.self_play(
        nac.compute_next_states, evaluator, nac.INITIAL_STATE, nac.utility,
        nac.which_player, nac.is_terminal, 1000, 1)

    assert len(action_probs_) == len(game_states_) - 1
    assert nac.is_terminal(game_states_[-1])
    assert game_states_[0] == nac.INITIAL_STATE
