
import numpy as np
import pytest

from alphago import mcts_tree


def test_mcts_tree_initial_tree():
    root = mcts_tree.MCTSNode(None, None)
    
    # Check the root has prior_prob None
    assert root.prior_prob is None
    assert root.Q == 0.0
    assert root.W == 0.0
    assert root.N == 0

    assert len(root.children) == 0

    assert root.game_state is None

def test_mcts_tree_is_leaf():
    leaf_node = mcts_tree.MCTSNode(None, None)
    assert mcts_tree.is_leaf(leaf_node)

def test_mcts_tree_expand_root():
    # Check we can expand the root of the tree.
    root = mcts_tree.MCTSNode(None, 1)

    children = {'a': 2, 'b': 3}
    prior_probs = {'a': 0.4, 'b': 0.6}

    leaf = root
    mcts_tree.expand(leaf, prior_probs, children)

    assert leaf.children['a'].game_state == 2
    assert leaf.children['a'].prior_prob == 0.4
    assert leaf.children['b'].game_state == 3
    assert leaf.children['b'].prior_prob == 0.6

def test_mcts_tree_selects_root_as_leaf():
    root = mcts_tree.MCTSNode(None, 1)

    nodes, actions = mcts_tree.select(root, 10, 10)
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
def test_mcts_backup(nodes, v):
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
    
@pytest.mark.parametrize("nodes, v, n", zip(backup_nodes_n_times, backup_values, backup_n))
def test_mcts_backup_twice_n_times(nodes, v, n):
    for i in range(n):
        mcts_tree.backup(nodes, v)
    for node in nodes:
        assert node.N == float(n)
        assert node.W == n*v
        assert node.Q == v
    
def test_mcts_select():
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

    nodes, actions = mcts_tree.select(root, 10, 1.0)
    # We should go through the root, then choose child 'b', then child 'e'.
    # This gives game states 1, 3, 6.
    assert [node.game_state for node in nodes] == [1, 3, 6]

def test_compute_ucb():
    c_puct = 1.0
    action_values = {'a': 1.0, 'b': 2.0, 'c': 3.0}
    prior_probs = {'a': 0.2, 'b': 0.5, 'c': 0.3}
    action_counts = {'a': 10, 'b': 20, 'c': 30}
    num = np.sqrt(sum(action_counts.values()))
    expected = {a: action_values[a] + prior_probs[a] / (1.0 + action_counts[a]) * c_puct * num for a in action_values}
    computed = mcts_tree.compute_ucb(action_values, prior_probs, action_counts, c_puct)
    assert expected == computed
