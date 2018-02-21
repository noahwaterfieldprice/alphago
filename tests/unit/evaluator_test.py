import numpy as np
import pytest
import tensorflow as tf

from alphago.evaluator import trivial_evaluator, BasicNACNet
from alphago import mcts_tree


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


@pytest.mark.parametrize("num_iters, expected", [
    (100, {0: 98.0 / 99.0, 1: 1.0 / 99.0}),
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


def test_initialising_basic_net_with_random_parameters():
    nnet = BasicNACNet()
    tensor_dict = nnet._initialise_net()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tensor_dict['loss'],
                 feed_dict={tensor_dict['state_vector']: np.random.rand(9),
                            tensor_dict['pi']: np.random.rand(1, 9),
                            tensor_dict['outcomes']: np.random.rand(1, 1)})


def test_neural_net_evaluator():
    def is_terminal(state):
        return len(next_states_function(state)) == 0

    def next_states_function_(state):
        return {action: (s,) * 9
                for action, s in next_states_function(state).items()}

    nnet = BasicNACNet()

    root = mcts_tree.MCTSNode(None, 0)
    action_probs = mcts_tree.mcts(
        root, nnet.evaluate, next_states_function_, is_terminal,
        max_iters=2, c_puct=1.0)
