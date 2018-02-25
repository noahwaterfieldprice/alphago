import numpy as np
import pytest

import alphago.games.noughts_and_crosses as nac
from alphago import mcts_tree
from alphago.evaluator import trivial_evaluator


def test_can_create_mcts_node():
    prior_prob = None
    game_state = None
    node = mcts_tree.MCTSNode(prior_prob, game_state, player=1)
    assert node is not None


def test_can_create_mcts_tree():
    game_state = None
    tree = mcts_tree.MCTSNode(None, game_state, player=1)
    assert tree is not None


# The dummy game is defined by a function that takes a state to a dictionary
# from actions available in state s to the state s.a (resulting from taking
# action a in state s).
def next_states_function(state):
    if state == 0:
        return {0: 1, 1: 2}
    elif state == 1:
        return {0: 3, 1: 4}
    elif state == 2:
        return {0: 5, 1: 6}
    else:
        return {}


def next_states_function_2(state):
    if state == 0:
        return {0: 1, 1: 2}
    elif state == 1:
        return {0: 3, 1: 4}
    elif state == 2:
        return {0: 5, 1: 6}
    elif state == 3:
        return {0: 7, 1: 8}
    elif state == 4:
        return {0: 9, 1: 10}
    else:
        return {}


def fake_which_player(state):
    return 1 + (int(np.ceil(state/2)) % 2)


def fake_utility(state):
    return {1: state, 2: -state}


# The evaluator returns a probs dictionary and scalar value. The probs
# dictionary has keys the actions a and values the probability that the
# evaluator assigns to the child state s.a. The value is the value that the
# evaluator assigns to the state s (note that it is the value for s and not
# s.a).
def evaluator_1(state):
    probs = {0: 0.5, 1: 0.5}
    value = 1.0
    return probs, value


def evaluator_2(state):
    probs = {0: 0.5, 1: 0.5}
    value = state
    return probs, value


@pytest.mark.parametrize(
    "next_states_function, evaluator, num_iters", [
        (next_states_function, evaluator_1, 100),
        (next_states_function, evaluator_1, 2),
    ]
)
def test_can_run_mcts_on_dummy_game(next_states_function, evaluator, num_iters):
    """ This test shows that we can run MCTS using a 'next_states' function and
    'evaluator' function.
    """
    def fake_is_terminal(state):
        return len(next_states_function(state)) == 0

    root = mcts_tree.MCTSNode(None, 0, player=1)
    action_probs = mcts_tree.mcts(
        root, evaluator, next_states_function, fake_utility, fake_which_player,
        fake_is_terminal, num_iters, 1.0
    )
    assert action_probs is not None


@pytest.mark.parametrize(
    "next_states_function, evaluator, max_iters, c_puct, expected", [
        (next_states_function, evaluator_1, 100, 1.0, [0, 1, 3]),
        (next_states_function_2, evaluator_1, 5, 1.0, [0, 1, 3, 8]),
    ]
)
def test_mcts_can_play_fake_game(next_states_function, evaluator, max_iters,
                                 c_puct, expected):
    # TODO: Check that expected [0, 1, 3, 8] in the second test is correct.
    def fake_is_terminal(state):
        return len(next_states_function(state)) == 0

    root = mcts_tree.MCTSNode(None, 0, player=1)
    node = root
    nodes = [node]

    print("Using next states function: {}".format(next_states_function))

    while len(next_states_function(node.game_state)) > 0:
        action_probs = mcts_tree.mcts(
            root, evaluator, next_states_function, fake_utility,
            fake_which_player, fake_is_terminal, max_iters, c_puct
        )
        action = max(action_probs, key=action_probs.get)
        print("Node: {}".format(node.game_state))
        print("Action probs: {}".format(action_probs))
        print("Node children: {}".format(node.children))
        print("Next states function(node): "
              "{}".format(next_states_function(node.game_state)))
        node = node.children[action]
        nodes.append(node)
    assert [node.game_state for node in nodes] == expected


@pytest.mark.parametrize(
    "next_states_function, evaluator, initial_state, max_iters,\
     c_puct, expected_length", [
        (next_states_function_2, evaluator_1, 0, 5, 1.0, 3),
        (next_states_function, evaluator_1, 0, 1000, 1.0, 3),
    ]
)
def test_mcts_can_self_play_fake_game(next_states_function, evaluator,
                                      initial_state, max_iters,
                                      c_puct, expected_length):
    def fake_is_terminal(state):
        return len(next_states_function(state)) == 0

    states, action_probs = mcts_tree.self_play(
        next_states_function, evaluator, initial_state, fake_utility,
        fake_which_player, fake_is_terminal, max_iters, c_puct
    )
    assert states[0] == initial_state
    assert len(states) == expected_length
    assert len(action_probs) == expected_length - 1


def test_mcts_can_self_play_noughts_and_crosses():
    max_iters = 1000
    c_puct = 1.0

    action_space = [(i, j) for i in range(3) for j in range(3)]

    def evaluator(state):
        return trivial_evaluator(
            state, nac.compute_next_states, action_space, nac.is_terminal,
            nac.utility, nac.which_player)

    game_states_, action_probs_ = mcts_tree.self_play(
        nac.compute_next_states, evaluator, nac.INITIAL_STATE, nac.utility,
        nac.which_player, nac.is_terminal, max_iters, c_puct
    )
    assert len(action_probs_) == len(game_states_) - 1
    assert nac.is_terminal(game_states_[-1])
    assert game_states_[0] == nac.INITIAL_STATE
