import numpy as np
import pytest
import torch

from alphago import MCTSNode, mcts
from alphago.estimator import (
    ConnectFourNet,
    NAC3x6NetEstimator,
    NACNetEstimator,
    create_trivial_estimator,
)
from alphago.games import ConnectFour, NoughtsAndCrosses

from .games.mock_game import MockGame
from .mock_estimator import MockNetEstimator

# TODO: Mock stuff properly in these tests


def test_trivial_estimator():
    mock_game = MockGame()
    trivial_estimator = create_trivial_estimator(mock_game)

    assert trivial_estimator(5) == ({0: 1 / 3, 1: 1 / 3, 2: 1 / 3}, 0)


def test_initialising_basic_net_with_random_parameters():
    nac = NoughtsAndCrosses()
    nnet = NACNetEstimator(
        learning_rate=0.01, l2_weight=0.1, action_indices=nac.action_indices
    )

    # A forward pass on a batch of all-ones states should run without error.
    states = torch.ones((7, 1, 3, 3))
    policy_logits, value = nnet.net(states)

    assert policy_logits.shape == (7, 9)
    assert value.shape == (7, 1)


def test_neural_net_estimator():
    mock_game = MockGame()
    nnet = MockNetEstimator(learning_rate=0.01)

    root = MCTSNode(0, player=1)
    mcts(root, mock_game, nnet, 100, 1.0)


def test_neural_net_estimate_game_state():
    nac = NoughtsAndCrosses()
    nnet = NACNetEstimator(
        learning_rate=0.01, l2_weight=0.1, action_indices=nac.action_indices
    )

    test_game_state = np.random.randn(7, 9)

    nnet(test_game_state)


def test_can_use_two_neural_nets():
    np.random.seed(0)
    torch.manual_seed(0)
    nac = NoughtsAndCrosses()
    nnet1 = NACNetEstimator(
        learning_rate=0.01, l2_weight=0.1, action_indices=nac.action_indices
    )
    nnet2 = NACNetEstimator(
        learning_rate=0.01, l2_weight=0.1, action_indices=nac.action_indices
    )

    test_game_state = np.random.randn(1, 9)

    probs_dict1, value1 = nnet1(test_game_state)
    probs_dict2, value2 = nnet2(test_game_state)

    # Check that the outputs are different. Since the input to both nets is the
    # same, this tests whether the nets are different.
    assert probs_dict1 != probs_dict2
    assert value1 != value2


def test_basic_nac_net_tensor_shapes():
    torch.manual_seed(0)
    nac = NoughtsAndCrosses()
    nnet = NACNetEstimator(
        learning_rate=0.01, l2_weight=0.1, action_indices=nac.action_indices
    )

    batch_size = 5
    states = torch.randn(batch_size, 1, 3, 3)

    policy_logits, value = nnet.net(states)

    assert policy_logits.shape == (batch_size, 9)
    assert value.shape == (batch_size, 1)


def test_forward_returns_logits_and_value():
    game = ConnectFour()
    net = ConnectFourNet(
        learning_rate=1e-4, l2_weight=1e-4, action_indices=game.action_indices
    )

    batch_size = 3
    states = torch.randn(batch_size, 1, 6, 7)

    out = net.net(states)
    assert isinstance(out, tuple)
    policy_logits, value = out
    assert policy_logits.shape == (batch_size, 7)
    assert value.shape == (batch_size, 1)
    # The value head is tanh, so it must lie in [-1, 1].
    assert torch.all(value >= -1) and torch.all(value <= 1)


def test_nac3x6_conv2d_shape():
    # Conv2d on the true rectangular (N, 2, 3, 6) board, and a
    # transposed (N, 2, 6, 3) input must fail loudly.
    action_indices = {i: i for i in range(18)}
    net = NAC3x6NetEstimator(learning_rate=0.01, action_indices=action_indices)

    batch_size = 4
    states = torch.randn(batch_size, 2, 3, 6)
    policy_logits, value = net.net(states)
    assert policy_logits.shape == (batch_size, 18)
    assert value.shape == (batch_size, 1)

    transposed = torch.randn(batch_size, 2, 6, 3)
    with pytest.raises(ValueError):
        net.net(transposed)


def test_nac_net_call():
    np.random.seed(0)
    torch.manual_seed(0)
    nac = NoughtsAndCrosses()
    net = NACNetEstimator(
        learning_rate=0.01, l2_weight=0.1, action_indices=nac.action_indices
    )

    state = (0,) * 9

    computed = net(state)

    probs_dict, value = computed
    assert isinstance(probs_dict, dict)
    assert len(probs_dict) == 9


def test_connect_four_net_runs_on_state():
    game = ConnectFour()
    net = ConnectFourNet(
        learning_rate=1e-4, l2_weight=1e-4, action_indices=game.action_indices
    )

    batch_size = 10
    states = torch.randn(batch_size, 1, 6, 7)

    policy_logits, value = net.net(states)
    assert policy_logits.shape == (batch_size, 7)
    assert value.shape == (batch_size, 1)


def test_connect_four_net_call():
    game = ConnectFour()
    net = ConnectFourNet(
        learning_rate=1e-4, l2_weight=1e-4, action_indices=game.action_indices
    )

    state = (0,) * 42

    computed = net(state)

    probs, value = computed

    assert isinstance(probs, dict)
    assert len(probs) == 7


def test_nac3x6_net_call():
    # Exercise the frozen seam end-to-end for NAC3x6 through its binary
    # _state_to_vector, not just PolicyValueNet.forward.
    torch.manual_seed(0)
    action_indices = {i: i for i in range(18)}
    net = NAC3x6NetEstimator(learning_rate=0.01, action_indices=action_indices)

    state = (0, 0)
    probs, value = net(state)

    assert isinstance(probs, dict)
    assert len(probs) == 18
    assert isinstance(value, float)


def test_create_estimator_zero_arg():
    # Constructing each concrete net with only
    # learning_rate + action_indices (no l2_weight) must not raise TypeError.
    nac = NoughtsAndCrosses()
    cf = ConnectFour()

    NACNetEstimator(learning_rate=0.01, action_indices=nac.action_indices)
    NAC3x6NetEstimator(learning_rate=0.01, action_indices={i: i for i in range(18)})
    ConnectFourNet(learning_rate=1e-4, action_indices=cf.action_indices)


def test_save_restore_roundtrip(tmp_path):
    # The .pt bundle round-trips weights + global_step and preserves
    # forward outputs.
    torch.manual_seed(0)
    game = ConnectFour()
    net = ConnectFourNet(
        learning_rate=1e-2, l2_weight=1e-4, action_indices=game.action_indices
    )

    # Take a training step so global_step and weights are non-trivial.
    state = (0,) * 42
    pi = np.array([1 / 7] * 7)
    net.train_step([(state, pi, 1.0)])
    assert net.global_step == 1

    before_probs, before_value = net(state)

    save_file = tmp_path / "checkpoint.pt"
    net.save(str(save_file))

    # Mutate the live net so a successful restore is observable.
    net.train_step([(state, pi, -1.0)])
    net.train_step([(state, pi, -1.0)])
    mutated_probs, _ = net(state)
    assert mutated_probs != before_probs

    net.restore(str(save_file))

    assert net.global_step == 1
    after_probs, after_value = net(state)
    assert after_value == pytest.approx(before_value)
    for action in before_probs:
        assert after_probs[action] == pytest.approx(before_probs[action])


def test_loss_non_overlapping_batches():
    # Each data row is used at most once across batches.
    from alphago.estimator import _iter_batches

    data = list(range(10))
    batch_size = 4

    seen = []
    for batch in _iter_batches(data, batch_size):
        assert len(batch) == batch_size
        seen.extend(batch)

    # Two full non-overlapping batches: [0..3], [4..7]; row 8, 9 dropped (no
    # partial batch), and crucially no row appears twice.
    assert len(seen) == len(set(seen))
    assert seen == [0, 1, 2, 3, 4, 5, 6, 7]
