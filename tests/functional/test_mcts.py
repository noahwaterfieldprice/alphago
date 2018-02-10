import numpy as np
import pytest

from alphago.mcts_tree import MCTSNode, MCTSTree


def test_can_create_mcts_node():
    prior_prob = None
    game_state = None
    node = MCTSNode(prior_prob, game_state)
    assert node is not None

def test_can_create_mcts_tree():
    game_state = None
    tree = MCTSTree(game_state)
    assert tree is not None

