import numpy as np

from alphago import mcts, MCTSNode
from alphago.estimator import create_trivial_estimator, NACNetEstimator
from alphago.games import NoughtsAndCrosses

from .games.mock_game import MockGame
from .mock_estimator import MockNetEstimator

# TODO: Mock stuff properly in these tests


def test_trivial_estimator():
    mock_game = MockGame()
    trivial_estimator = create_trivial_estimator(mock_game.compute_next_states)

    assert trivial_estimator(5) == ({0: 1 / 3, 1: 1 / 3, 2: 1 / 3}, 0)


def test_initialising_basic_net_with_random_parameters():  # TODO: redo this on mock game
    nac = NoughtsAndCrosses()
    nnet = NACNetEstimator(learning_rate=0.01, action_indices=nac.action_indices)

    # Initialise state of all 1s.
    states = np.ones((7, 9))
    pis = np.random.rand(7, 9)
    outcomes = np.random.rand(7, 1)

    nnet.sess.run(nnet.tensors['loss'],
                  feed_dict={nnet.tensors['state_vector']: states,
                             nnet.tensors['pi']: pis,
                             nnet.tensors['outcomes']: outcomes})


def test_neural_net_estimator():
    mock_game = MockGame()
    nnet = MockNetEstimator(learning_rate=0.01)

    root = MCTSNode(0, player=1)
    action_probs = mcts(root, mock_game, nnet, 100, 1.0)


def test_neural_net_estimate_game_state():
    nac = NoughtsAndCrosses()
    nnet = NACNetEstimator(learning_rate=0.01, action_indices=nac.action_indices)

    test_game_state = np.random.randn(7, 9)

    computed = nnet(test_game_state)


def test_can_use_two_neural_nets():
    np.random.seed(0)
    nac = NoughtsAndCrosses()
    nnet1 = NACNetEstimator(learning_rate=0.01, action_indices=nac.action_indices)
    nnet2 = NACNetEstimator(learning_rate=0.01, action_indices=nac.action_indices)

    test_game_state = np.random.randn(7, 9)

    computed1 = nnet1(test_game_state)
    computed2 = nnet2(test_game_state)

    # Check that the outputs are different. Since the input to both nets is the
    # same, this tests whether the nets are different.
    assert not (computed1[0] == computed2[0]).all()
    assert not (computed1[1] == computed2[1]).all()


def test_basic_nac_net_tensor_shapes():
    np.random.seed(0)
    nac = NoughtsAndCrosses()
    nnet = NACNetEstimator(learning_rate=0.01, action_indices=nac.action_indices)

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
        nnet.tensors['value'],
    ]

    computed_tensors = nnet.sess.run(
        tensors, feed_dict={
            nnet.tensors['state_vector']: states,
            nnet.tensors['pi']: pis,
            nnet.tensors['outcomes']: zs,
            })

    loss, loss_probs, loss_value, probs, value = computed_tensors

    # The loss should be positive
    assert loss_probs > 0
    assert loss_value > 0
    assert loss > 0

    # The loss should be a scalar.
    assert np.shape(loss) == ()
    assert np.shape(probs) == (batch_size, 9)
    assert np.shape(value) == (batch_size, 1)
