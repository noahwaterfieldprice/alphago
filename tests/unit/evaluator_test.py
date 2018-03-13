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

    nnet.sess.run(tensor_dict['loss'],
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


def test_can_use_two_neural_nets():
    np.random.seed(0)
    nnet1 = BasicNACNet()
    nnet2 = BasicNACNet()

    test_game_state = np.random.randn(7, 9)

    computed1 = nnet1.evaluate(test_game_state)
    computed2 = nnet2.evaluate(test_game_state)

    # Check that the outputs are different. Since the input to both nets is the
    # same, this tests whether the nets are different.
    assert not (computed1[0] == computed2[0]).all()
    assert not (computed1[1] == computed2[1]).all()


def test_basic_nac_net_tensor_shapes():
    np.random.seed(0)
    nnet = BasicNACNet()

    batch_size = 5

    # Set up the states, probs, zs arrays.
    states = np.random.randn(batch_size, 9)
    pis = np.random.rand(batch_size, 9)
    zs = np.random.randn(batch_size, 1)

    tensors = [
        nnet.tensors['loss'],
        nnet.tensors['loss_probs'],
        nnet.tensors['loss_value'],
        nnet.tensors['probs'],
        nnet.tensors['values'],
    ]

    computed_tensors = nnet.sess.run(
        tensors, feed_dict={
            nnet.tensors['state_vector']: states,
            nnet.tensors['pi']: pis,
            nnet.tensors['outcomes']: zs,
            })
    loss = computed_tensors[0]
    loss_probs = computed_tensors[1]
    loss_value = computed_tensors[2]
    probs = computed_tensors[3]
    values = computed_tensors[4]

    # The loss should be positive
    assert loss_probs > 0
    assert loss_value > 0
    assert loss > 0

    # The loss should be a scalar.
    assert np.shape(loss) == ()
    assert np.shape(probs) == (batch_size, 9)
    assert np.shape(values) == (batch_size, 1)
