import pytest

from alphago import MCTSNode, NoughtsAndCrosses, mcts
from alphago.estimator import create_trivial_estimator
from alphago.mcts_tree import print_tree

from ..unit.games.mock_game import MockGame


def test_can_create_mcts_node():
    game_state = None
    node = MCTSNode(game_state, player=1)
    assert node is not None


def test_can_create_mcts_tree():
    game_state = None
    tree = MCTSNode(game_state, player=1)
    assert tree is not None


def test_can_run_mcts_on_fake_game():
    """This test shows that we can run MCTS using a 'next_states'
    function and 'evaluator' function.
    """
    mock_game = MockGame()
    root = MCTSNode(0, player=1)
    action_probs = mcts(root, mock_game, mock_game.mock_estimator, 100, 1.0)

    assert action_probs is not None


def mock_evaluator_action0(state):
    prior_probs = {
        action: prob for action, prob in zip(range(3), [1, 0, 0], strict=False)
    }
    value = 0
    return prior_probs, value


def mock_evaluator_action1(state):
    prior_probs = {
        action: prob for action, prob in zip(range(3), [0, 1, 0], strict=False)
    }
    value = 0
    return prior_probs, value


@pytest.mark.parametrize(
    "evaluator, expected",
    [
        (mock_evaluator_action0, [0, 1, 3, 7]),
        (mock_evaluator_action1, [0, 2, 6, 17]),
    ],
)
def test_mcts_can_play_fake_game(evaluator, expected):
    mock_game = MockGame()

    root = MCTSNode(0, player=1)
    node = root
    nodes = [node]

    while not node.is_terminal:
        action_probs = mcts(root, mock_game, evaluator, 100, 1)

        action = max(action_probs, key=action_probs.get)
        node = node.children[action]
        nodes.append(node)
    assert [node.game_state for node in nodes] == expected


def test_print_mcts():
    mock_game = MockGame()
    mock_game.terminal_state_values = tuple(0.01 * i for i in range(12))

    root = MCTSNode(0, player=1)
    mcts(root, mock_game, mock_game.mock_estimator, 10, 1.0)
    print_tree(root)

    assert root.N == 10


def test_mcts_returns_valid_visit_distribution_over_legal_actions():
    """mcts() returns a well-formed visit distribution.

    Over a real game with the uniform trivial estimator, the returned
    distribution must be shape-correct (one entry per child), sum to 1,
    be non-negative, and have keys that are a subset of legal actions.
    """
    nac = NoughtsAndCrosses()
    estimator = create_trivial_estimator(nac)
    root = MCTSNode(nac.initial_state, player=1)

    probs = mcts(
        root,
        nac,
        estimator,
        mcts_iters=100,
        c_puct=1.0,
        dirichlet_epsilon=0.0,
    )

    legal = set(nac.legal_actions(nac.initial_state).keys())
    assert set(probs.keys()) <= legal
    assert len(probs) == len(root.children)
    assert sum(probs.values()) == pytest.approx(1.0)
    assert all(p >= 0 for p in probs.values())
