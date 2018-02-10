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


def test_can_run_mcts_on_dummy_game():
    """ This test shows that we can run MCTS using a 'next_states' function and
    'evaluator' function.
    """
    # The dummy game is defined by a function that takes a state to a dictionary
    # from actions available in state s to the state s.a (resulting from taking
    # action a in state s).
    def next_states(state):
        if state == 0:
            return {0: 1, 1: 2}
        elif state == 1:
            return {0: 3, 1: 4}
        elif state == 2:
            return {0: 5, 1: 6}
        else:
            return {}

    # The evaluator returns a probs dictionary and scalar value. The probs
    # dictionary has keys the actions a and values the probability that the
    # evaluator assigns to the child state s.a. The value is the value that the
    # evaluator assigns to the state s (note that it is the value for s and not
    # s.a).
    def evaluator(state):
        probs = {0: 0.5, 1: 0.5}
        value = state
        return probs, value

    root = mcts_tree.MCTSNode(None, 0)
    action_probs = mcts_tree.mcts(root, evaluator, next_states, 100, 10, 1.0)
    assert action_probs is not None
