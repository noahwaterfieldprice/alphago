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


def fake_which_player(state):
    return 1 + (int(np.ceil(state/2)) % 2)


def fake_utility(state):
    return {1: state, 2: -state}


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

    root = mcts_tree.MCTSNode(None, 0, player=1)
    action_probs = mcts_tree.mcts(
        root, evaluator, next_states_function, utility, which_player,
        is_terminal, num_iters, 1.0
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
    def next_states_function_(state):
        return {action: (s,) * 2
                for action, s in next_states_function(state[0]).items()}

    def fake_is_terminal(state):
        return len(next_states_function_(state)) == 0

    def fake_utility_(state):
        return fake_utility(state[0])

    def fake_which_player_(state):
        return fake_which_player(state[0])

    nnet = BasicNACNet(input_dim=2, output_dim=2)

    # TODO THE PROBLEM IS THAT THE GAME AS DESCRIBED DOESN'T WORK.

    root = mcts_tree.MCTSNode(None, (0,) * 2, player=1)
    # pytest.set_trace()
    action_probs = mcts_tree.mcts(
        root, nnet.evaluate, next_states_function_, fake_utility_,
        fake_which_player_, fake_is_terminal,
        max_iters=2, c_puct=1.0)


def test_neural_net_evaluate_game_state():
    nnet = BasicNACNet()

    test_game_state = (0,) * 9

    computed = nnet.evaluate(test_game_state)
