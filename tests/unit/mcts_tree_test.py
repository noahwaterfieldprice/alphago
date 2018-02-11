
import numpy as np
import pytest

from alphago import mcts_tree
from alphago.evaluator import trivial_evaluator


class TestMCTSNode:
    def test_mcts_tree_initial_tree(self):
        root = mcts_tree.MCTSNode(None, None)

        # Check the root has prior_prob None
        assert root.prior_prob is None
        assert root.Q == 0.0
        assert root.W == 0.0
        assert root.N == 0

        assert len(root.children) == 0

        assert root.game_state is None

    def test_mcts_tree_is_leaf(self):
        leaf_node = mcts_tree.MCTSNode(None, None)
        assert leaf_node.is_leaf()

    def test_mcts_tree_expand_root(self):
        # Check we can expand the root of the tree.
        root = mcts_tree.MCTSNode(None, 1)

        children_states = {'a': 2, 'b': 3}
        prior_probs = {'a': 0.4, 'b': 0.6}

        leaf = root
        leaf.expand(prior_probs, children_states)

        assert leaf.children['a'].game_state == 2
        assert leaf.children['a'].prior_prob == 0.4
        assert leaf.children['b'].game_state == 3
        assert leaf.children['b'].prior_prob == 0.6


class TestSelectAndBackupFunctions:
    def test_mcts_tree_selects_root_as_leaf(self):
        root = mcts_tree.MCTSNode(None, 1)

        nodes, actions = mcts_tree.select(root, 10)
        assert len(actions) == 0
        assert len(nodes) == 1
        assert nodes[0] == root

    backup_nodes = [
        [mcts_tree.MCTSNode(None, 3)],
        [mcts_tree.MCTSNode(None, None), mcts_tree.MCTSNode(None, None)],
        [mcts_tree.MCTSNode({'a': 1.0}, 1), mcts_tree.MCTSNode({3: 0.5}, 2)],

    ]

    backup_values = [
        1.0,
        2.0,
        3.0,
    ]

    @pytest.mark.parametrize("nodes, v", zip(backup_nodes, backup_values))
    def test_mcts_backup(self, nodes, v):
        mcts_tree.backup(nodes, v)
        for node in nodes:
            assert node.N == 1.0
            assert node.W == v
            assert node.Q == v

    backup_nodes_n_times = [
        [mcts_tree.MCTSNode(None, 3)],
        [mcts_tree.MCTSNode(None, None), mcts_tree.MCTSNode(None, None)],
        [mcts_tree.MCTSNode({'a': 1.0}, 1), mcts_tree.MCTSNode({3: 0.5}, 2)],
    ]

    backup_n = [
        1,
        2,
        3,
    ]

    @pytest.mark.parametrize("nodes, v, n",
                             zip(backup_nodes_n_times, backup_values, backup_n))
    def test_mcts_backup_twice_n_times(self, nodes, v, n):
        for i in range(n):
            mcts_tree.backup(nodes, v)
        for node in nodes:
            assert node.N == float(n)
            assert node.W == n*v
            assert node.Q == v

    def test_mcts_select(self):
        root = mcts_tree.MCTSNode(None, 1)

        # Manually create a small tree below root.
        root.children = {'a': mcts_tree.MCTSNode(0.2, 2), 'b': mcts_tree.MCTSNode(0.8, 3)}
        root.prior_prob = 1.0
        root.N = 4
        childa = root.children['a']
        childa.children = {'c': mcts_tree.MCTSNode(0.7, 4), 'd': mcts_tree.MCTSNode(0.3, 5)}
        childa.N = 2

        nodec = childa.children['c']
        nodec.N = 1
        noded = childa.children['d']
        noded.N = 1

        childb = root.children['b']
        childb.children = {'e': mcts_tree.MCTSNode(0.9, 6), 'f': mcts_tree.MCTSNode(0.1, 7)}
        childb.N = 1

        nodee = childb.children['e']
        nodee.N = 1
        nodef = childb.children['f']
        nodef.N = 1

        nodes, actions = mcts_tree.select(root, 1.0)
        # We should go through the root, then choose child 'b', then child 'e'.
        # This gives game states 1, 3, 6.
        assert [node.game_state for node in nodes] == [1, 3, 6]


def test_compute_ucb():
    c_puct = 1.0
    action_values = {'a': 1.0, 'b': 2.0, 'c': 3.0}
    prior_probs = {'a': 0.2, 'b': 0.5, 'c': 0.3}
    action_counts = {'a': 10, 'b': 20, 'c': 30}
    num = np.sqrt(sum(action_counts.values()))
    expected = {
        a: action_values[a] + prior_probs[a] / (1.0 + action_counts[a]) *
        c_puct * num for a in action_values
    }
    computed = mcts_tree.compute_ucb(
        action_values,
        prior_probs,
        action_counts,
        c_puct)
    assert expected == computed


def test_compute_distribution():
    action_counts = {1: 3, 5: 7}
    computed = mcts_tree.compute_distribution(action_counts)
    expected = {1: 3.0 / 10.0, 5: 7.0 / 10.0}
    assert computed == expected


# We first create a dummy game
def next_states_function(state):
    if state == 0:
        return {0: 1, 1: 2}
    elif state == 1:
        return {0: 3, 1: 4}
    elif state == 2:
        return {0: 5, 1: 6}
    else:
        return {}


def evaluator_1(state):
    probs = {0: 0.5, 1: 0.5}
    value = 1.0
    return probs, value


def evaluator_2(state):
    probs = {0: 0.5, 1: 0.5}
    value = state
    return probs, value


@pytest.mark.parametrize(
    "next_states_function, evaluator", [
        (next_states_function, evaluator_1),
        (next_states_function, evaluator_2),
    ]
)
def test_mcts_action_count_at_root(next_states_function, evaluator):
    root = mcts_tree.MCTSNode(None, 0)
    assert root.N == 0

    def is_terminal(state):
        return len(next_states_function(state)) == 0

    action_probs = mcts_tree.mcts(
        root, evaluator, next_states_function, is_terminal, 100, 1.0
    )

    # Each iteration of MCTS we should add 1 to N at the root.
    assert root.N == 100


@pytest.mark.parametrize("evaluator, num_iters, expected", [
        (evaluator_1, 100, 100),
        (evaluator_1, 2, 2),
    ]
)
def test_mcts_value_at_root(evaluator, num_iters, expected):

    root = mcts_tree.MCTSNode(None, 0)
    assert root.N == 0

    def is_terminal(state):
        return len(next_states_function(state)) == 0

    action_probs = mcts_tree.mcts(
        root, evaluator, next_states_function, is_terminal, num_iters, 1.0
    )

    # Each iteration of MCTS we should add 1 to N at the root.
    assert root.W == expected


@pytest.mark.parametrize(
    "evaluator, num_iters, expected", [
        (evaluator_1, 100, 100),
        (evaluator_1, 2, 2),
    ]
)
def test_mcts_does_not_expand_terminal_nodes(evaluator, num_iters, expected):
    def is_terminal(state):
        return len(next_states_function(state)) == 0

    def next_states_wrapper(state):
        assert not is_terminal(state)
        return next_states_function(state)

    root = mcts_tree.MCTSNode(None, 0)
    action_probs = mcts_tree.mcts(
        root, evaluator, next_states_wrapper, is_terminal, num_iters, 1.0
    )


@pytest.mark.parametrize("num_iters, expected", [
        (100, {0: 98.0/99.0, 1: 1.0/99.0}),
        (2, {0: 1.0, 1: 0.0}),
    ]
)
def test_trivial_evaluator(num_iters, expected):

    def is_terminal(state):
        return len(next_states_function(state)) == 0

    def utility(state):
        return {1: state, 2: state}

    def which_player(state):
        return 1

    action_space = [0, 1]

    def evaluator(state):
        return trivial_evaluator(
            state, next_states_function, action_space, is_terminal,
            utility, which_player)

    root = mcts_tree.MCTSNode(None, 0)
    action_probs = mcts_tree.mcts(
        root, evaluator, next_states_function, is_terminal, num_iters, 1.0
    )
    assert action_probs == expected


TRAINING_DATA_STATES = [
    [1, 2, 3, 4],
    [1, 4, 3, 6, 7],
]

TRAINING_DATA_ACTION_PROBS = [
    [{1: 0.5, 2: 0.5}, {3: 0.7}, {2: 0.3, 5: 0.7}],
    [{1: 0.5, 2: 0.5}, {3: 0.7}, {2: 0.3, 5: 0.7}, {1: 1.0}],
]

TRAINING_DATA_EXPECTED = [
    [(1, {1: 0.5, 2: 0.5}, -4), (2, {3: 0.7}, 4), (3, {2: 0.3, 5: 0.7}, -4)],
    [(1, {1: 0.5, 2: 0.5}, -7), (4, {3: 0.7}, 7), (3, {2: 0.3, 5: 0.7}, -7),
        (6, {1: 1.0}, 7)],
]


@pytest.mark.parametrize("states_, action_probs_, expected",
                         zip(TRAINING_DATA_STATES, TRAINING_DATA_ACTION_PROBS,
                             TRAINING_DATA_EXPECTED))
def test_build_training_data(states_, action_probs_, expected):

    def which_player(state):
        return 1 + (state % 2)

    def utility(state):
        return Outcome(state, -state)

    training_data = mcts_tree.build_training_data(states_, action_probs_,
                                                  which_player, utility)

    assert training_data == expected
