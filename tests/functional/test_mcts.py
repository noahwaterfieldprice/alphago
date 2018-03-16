import pytest

from alphago import mcts, MCTSNode
from alphago.mcts_tree import print_tree
from ..unit.games import mock_game


def test_can_create_mcts_node():
    game_state = None
    node = MCTSNode(game_state, player=1)
    assert node is not None


def test_can_create_mcts_tree():
    game_state = None
    tree = MCTSNode(game_state, player=1)
    assert tree is not None


def test_can_run_mcts_on_fake_game():
    """ This test shows that we can run MCTS using a 'next_states'
    function and 'evaluator' function.
    """
    root = MCTSNode(0, player=1)
    action_probs = mcts(root, mock_game.mock_evaluator, mock_game, 100, 1.0)

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
    root = MCTSNode(0, player=1)
    node = root
    nodes = [node]

    while not node.is_terminal:
        action_probs = mcts(root, evaluator, mock_game, 100, 1)

        action = max(action_probs, key=action_probs.get)
        node = node.children[action]
        nodes.append(node)
    assert [node.game_state for node in nodes] == expected


def test_print_mcts():
    # TODO: This doesn't currently test anything.
    # TODO: This next line is probably a bad idea
    mock_game.TERMINAL_STATE_VALUES = [0.01 * i for i in range(12)]

    root = MCTSNode(0, player=1)
    action_probs = mcts(root, mock_game.mock_evaluator, mock_game, 10, 1.0)
    print_tree(root)
    mock_game.TERMINAL_STATE_VALUES = (1,) * 12
