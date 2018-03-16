import numpy as np
import pytest

from alphago import mcts, MCTSNode
from alphago.mcts_tree import backup, compute_distribution, compute_ucb, select
from .games import mock_game


# Utility functions


def get_terminal_nodes(root):
    stack = [root]
    terminal_nodes = []
    while stack:
        node = stack.pop()
        if node.is_terminal:
            terminal_nodes.append(node)
        else:
            stack.extend(node.children.values())

    return terminal_nodes


class TestMCTSNode:
    def test_mcts_tree_initial_tree(self):
        root = MCTSNode(None, 1)

        assert root.Q == 0.0
        assert root.W == 0.0
        assert root.N == 0

        assert len(root.children) == 0
        assert len(root.prior_probs) == 0

        assert root.game_state is None

    def test_mcts_tree_is_leaf(self):
        leaf_node = MCTSNode(None, player=1)
        assert leaf_node.is_leaf()

    def test_mcts_tree_expand_root(self):
        # Check we can expand the root of the tree.
        root = MCTSNode(1, player=1)

        children_states = {'a': 2, 'b': 3}
        prior_probs = {'a': 0.4, 'b': 0.6}
        child_players = {'a': 2, 'b': 2}
        child_terminals = {'a': False, 'b': False}

        leaf = root
        leaf.expand(prior_probs, children_states, child_players, child_terminals)

        assert leaf.children['a'].game_state == 2
        assert leaf.children['b'].game_state == 3
        assert leaf.prior_probs == prior_probs


class TestSelectAndBackupFunctions:
    def test_mcts_tree_selects_root_as_leaf(self):
        root = MCTSNode(1, player=1)

        nodes, actions = select(root, 10)
        assert len(actions) == 0
        assert len(nodes) == 1
        assert nodes[0] == root

    backup_nodes = [
        [MCTSNode(3, player=1)],
        [MCTSNode(None, player=1), MCTSNode(None, player=1)],
        [MCTSNode(1, player=1), MCTSNode(2, player=2), MCTSNode(3, player=1)],
    ]

    backup_expected_Q = [
        [0.0],
        [0.0, 2.0],
        [0.0, 3.0, -3.0],
    ]

    backup_values = [
        {1: 1.0, 2: -1.0},
        {1: 2.0, 2: -2.0},
        {1: 3.0, 2: -3.0},
    ]

    @pytest.mark.parametrize("nodes, expected_Q, values",
                             zip(backup_nodes, backup_expected_Q,
                                 backup_values))
    def test_mcts_backup(self, nodes, expected_Q, values):
        backup(nodes, values)
        for i, node in enumerate(nodes):
            assert node.N == 1.0
            assert node.W == expected_Q[i]
            assert node.Q == expected_Q[i]

    backup_nodes_n_times = [
        [MCTSNode(3, player=1)],
        [MCTSNode(None, player=1), MCTSNode(None, player=1)],
        [MCTSNode(1, player=1), MCTSNode(2, player=2), MCTSNode(3, player=1)],
    ]

    backup_n = [1, 2, 3]

    @pytest.mark.parametrize("nodes, expected_Q, values, n",
                             zip(backup_nodes_n_times, backup_expected_Q,
                                 backup_values, backup_n))
    def test_mcts_backup_n_times(self, nodes, expected_Q, values, n):
        for i in range(n):
            backup(nodes, values)

        for i, node in enumerate(nodes):
            assert node.N == n
            assert node.W == n * expected_Q[i]
            assert node.Q == expected_Q[i]

    def test_mcts_select(self):
        root = MCTSNode(1, player=1)

        # Manually create a small tree below root.
        root.children = {'a': MCTSNode(2, player=2),
                         'b': MCTSNode(3, player=2)}
        root.prior_probs = {'a': 0.2, 'b': 0.8}
        root.N = 4
        childa = root.children['a']
        childa.children = {'c': MCTSNode(4, player=1),
                           'd': MCTSNode(5, player=1)}
        childa.prior_probs = {'c': 0.7, 'd': 0.3}
        childa.N = 2

        nodec = childa.children['c']
        nodec.N = 1
        noded = childa.children['d']
        noded.N = 1

        childb = root.children['b']
        childb.children = {'e': MCTSNode(6, player=1),
                           'f': MCTSNode(7, player=1)}
        childb.prior_probs = {'e': 0.9, 'f': 0.1}
        childb.N = 1

        nodee = childb.children['e']
        nodee.N = 1
        nodef = childb.children['f']
        nodef.N = 1

        nodes, actions = select(root, 1.0)
        # We should go through the root, then choose child 'b', then child 'e'.
        # This gives game states 1, 3, 6.
        assert [node.game_state for node in nodes] == [1, 3, 6]


def test_compute_ucb():
    c_puct = 1.0
    action_values = {'a': 1.0, 'b': 2.0, 'c': 3.0}
    prior_probs = {'a': 0.2, 'b': 0.5, 'c': 0.3}
    action_counts = {'a': 10, 'b': 20, 'c': 30}
    num = 1.0 + np.sqrt(sum(action_counts.values()))
    expected = {
        a: action_values[a] + prior_probs[a] / (1.0 + action_counts[a]) *
        c_puct * num for a in action_values
    }
    computed = compute_ucb(
        action_values,
        prior_probs,
        action_counts,
        c_puct)
    assert expected == computed


def test_compute_distribution():
    action_counts = {1: 3, 5: 7}
    computed = compute_distribution(action_counts)
    expected = {1: 3.0 / 10.0, 5: 7.0 / 10.0}
    assert computed == expected


def test_mcts_action_count_at_root():
    root = MCTSNode(0, player=1)
    assert root.N == 0

    action_probs = mcts(root, mock_game.mock_evaluator, mock_game, 100, 1.0)

    # Each iteration of MCTS we should add 1 to N at the root.
    assert root.N == 100


def test_mcts_action_count_at_root_children():
    root = MCTSNode(0, player=1)

    action_probs = mcts(
        root, mock_game.mock_evaluator, mock_game, 100, 1.0)

    # Each iteration of MCTS we should add 1 to N at the root.
    assert sum(child.N for child in root.children.values()) == 99


def test_mcts_value_at_children_of_root():
    root = MCTSNode(0, player=1)
    assert root.N == 0

    mcts(root, mock_game.mock_evaluator, mock_game, 100, 1.0)

    terminal_nodes = get_terminal_nodes(root)
    N_terminal_nodes = sum(node.N for node in terminal_nodes)

    # Each iteration of MCTS we should add 1 to W of one of the
    # children of the root.
    assert sum(child.W for child in root.children.values()) == N_terminal_nodes

# TODO: work out how to do this now mcts takes game object
# def test_mcts_does_not_expand_terminal_nodes():
#     mock_game = MockGame()
#
#     def next_states_wrapper(state):
#         assert not mock_game.is_terminal(state)
#         return mock_game.compute_next_states(state)
#
#
#
#     root = MCTSNode(0, player=1)
#
#     mcts(root, mock_evaluator, next_states_wrapper,
#                    mock_game.utility, mock_game.which_player,
#                    mock_game.is_terminal, 100, 1.0)
