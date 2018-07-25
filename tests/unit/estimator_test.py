import numpy as np

from alphago import mcts, MCTSNode
from alphago.estimator import (create_trivial_estimator, NACNetEstimator,
                               ConnectFourNet)
from alphago.games import NoughtsAndCrosses, ConnectFour

from .games.mock_game import MockGame
from .mock_estimator import MockNetEstimator

# TODO: Mock stuff properly in these tests


def test_trivial_estimator():
    mock_game = MockGame()
    trivial_estimator = create_trivial_estimator(mock_game)

    assert trivial_estimator(5) == ({0: 1 / 3, 1: 1 / 3, 2: 1 / 3}, 0)


def test_initialising_basic_net_with_random_parameters():  # TODO: redo this on mock game
    nac = NoughtsAndCrosses()
    nnet = NACNetEstimator(learning_rate=0.01, l2_weight=0.1,
                           action_indices=nac.action_indices)

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
    nnet = NACNetEstimator(learning_rate=0.01, l2_weight=0.1,
                           action_indices=nac.action_indices)

    test_game_state = np.random.randn(7, 9)

    computed = nnet(test_game_state)


def test_can_use_two_neural_nets():
    np.random.seed(0)
    nac = NoughtsAndCrosses()
    nnet1 = NACNetEstimator(learning_rate=0.01, l2_weight=0.1,
                            action_indices=nac.action_indices)
    nnet2 = NACNetEstimator(learning_rate=0.01, l2_weight=0.1,
                            action_indices=nac.action_indices)

    test_game_state = np.random.randn(1, 9)

    probs_dict1, value1 = nnet1(test_game_state)
    probs_dict2, value2 = nnet2(test_game_state)

    # Check that the outputs are different. Since the input to both nets is the
    # same, this tests whether the nets are different.
    assert probs_dict1 != probs_dict2
    assert value1 != value2


def test_basic_nac_net_tensor_shapes():
    np.random.seed(0)
    nac = NoughtsAndCrosses()
    nnet = NACNetEstimator(learning_rate=0.01, l2_weight=0.1,
                           action_indices=nac.action_indices)

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


def test_nac_net_call():
    np.random.seed(0)
    nac = NoughtsAndCrosses()
    net = NACNetEstimator(learning_rate=0.01, l2_weight=0.1,
                          action_indices=nac.action_indices)

    state = (0,) * 9

    computed = net(state)

    probs_dict, value = computed
    assert isinstance(probs_dict, dict)
    assert len(probs_dict) == 9


def test_connect_four_net_runs_on_state():
    game = ConnectFour()
    net = ConnectFourNet(learning_rate=1e-4, l2_weight=1e-4,
                         action_indices=game.action_indices)

    batch_size = 10
    states = np.random.randn(batch_size, 42)

    assert np.shape(states) == (10, 42)

    tensors = [
        net.tensors['probs'],
        net.tensors['value'],
    ]

    computed = net.sess.run(
        tensors, feed_dict={
            net.tensors['state_vector']: states
        }
    )

    probs, value = computed
    assert np.shape(probs) == (batch_size, 7)
    assert np.shape(value) == (batch_size, 1)


def test_connect_four_net_call():
    game = ConnectFour()
    net = ConnectFourNet(learning_rate=1e-4, l2_weight=1e-4,
                         action_indices=game.action_indices)

    state = (0,) * 42

    computed = net(state)

    probs, value = computed

    assert isinstance(probs, dict)
    assert len(probs) == 7
