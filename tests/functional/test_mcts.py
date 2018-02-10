import numpy as np
import pytest

from alphago import mcts_tree


def test_can_create_mcts_node():
    prior_prob = None
    game_state = None
    node = mcts_tree.MCTSNode(prior_prob, game_state)
    assert node is not None


def test_can_create_mcts_tree():
    game_state = None
    tree = mcts_tree.MCTSNode(None, game_state)
    assert tree is not None
