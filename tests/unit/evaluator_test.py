import numpy as np
import pytest
import tensorflow as tf

from alphago.evaluator import create_trivial_evaluator, BasicNACNet
from alphago import mcts_tree
from .games.mock_game import MockGame, mock_evaluator
from .mock_evaluator import MockNet


def test_trivial_evaluator():
    mock_game = MockGame()
    trivial_evaluator = create_trivial_evaluator(mock_game.compute_next_states)

    assert trivial_evaluator(5) == ({0: 1 / 3, 1: 1 / 3, 2: 1 / 3}, 0)

    # action_probs = mcts_tree.mcts(
    #     root, trivial_evaluator, mock_game.compute_next_states,
    #     mock_game.utility, mock_game.which_player,
    #     mock_game.is_terminal, 100, 1.0)
    # assert action_probs == 0


def test_initialising_basic_net_with_random_parameters():
    nnet = BasicNACNet()
    tensor_dict = nnet._initialise_net()

    # Initialise state of all 1s.
    states = np.ones((7, 9))
    pis = np.random.rand(7, 9)
    outcomes = np.random.rand(7, 1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tensor_dict['loss'],
                 feed_dict={tensor_dict['state_vector']: states,
                            tensor_dict['pi']: pis,
                            tensor_dict['outcomes']: outcomes})


def test_neural_net_evaluator():
    mock_game = MockGame()
    nnet = MockNet(input_dim=1, output_dim=3)

    root = mcts_tree.MCTSNode(0, player=1)
    action_probs = mcts_tree.mcts(
        root, nnet.evaluate, mock_game.compute_next_states,
        mock_game.utility, mock_game.which_player,
        mock_game.is_terminal, 100, 1.0)


def test_neural_net_evaluate_game_state():
    nnet = BasicNACNet()

    test_game_state = np.random.randn(7, 9)

    computed = nnet.evaluate(test_game_state)
