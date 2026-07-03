"""Policy/value estimators for the AlphaGo Zero networks.

This module ports the three game-specific neural-net estimators from
TensorFlow 1.x to ``torch.nn``. The three near-identical TF topologies are
collapsed into a single configurable :class:`PolicyValueNet` driven by a
per-game :class:`NetConfig`. The frozen estimator seam is
preserved verbatim: a callable ``(state) -> (probs_dict, value)`` keyed by all
actions, a ``create_estimator()``-style zero-arg factory, and
``train``/``save``/``restore``.

Three documented behaviour changes (NOT faithful reproductions of the TF1 code)
are folded in so later regressions are attributable:

* **Explicit L2.** The TF1 code registered ``weights_regularizer`` on every
  layer but never added it to the minimized loss (old ``estimator.py:465``), so
  TF1 trained effectively L2-free. The port adds an explicit L2 penalty to the
  loss.
* **Dropping BatchNorm.** ``NAC3x6NetEstimator`` had ``use_batch_norm=True``
  ACTIVE (old ``estimator.py:689``); removing BN is a real behaviour change for
  NAC3x6 (a no-op only for NAC3x3 and Connect Four).
* **l2_weight default.** The concrete TF1 ``__init__`` declared ``l2_weight``
  with no default (old ``estimator.py:365/523/876``), so the zero-arg
  ``create_estimator()`` factory raised ``TypeError``. ``l2_weight`` now has a
  sane default.

The 3D-conv -> Conv2d fix (on the true ``(N, C, H, W)`` board) and
non-overlapping minibatches in :meth:`loss` are also folded in.
"""

import abc
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .device import get_device
from .games import Game


def create_trivial_estimator(game: Game):
    """Create a trivial evaluator function given a next states function
    for a game.

    Parameters
    ----------
    game:


    Returns
    -------
    trivial_estimator: func
        A function that returns a uniform probability distribution over all
        possible actions and a value given an input state.
    """

    def trivial_estimator(state):
        """Evaluates a game state for a game. It is trivial in the sense
        that it returns the uniform probability distribution over all
        actions in the game.

        Parameters
        ----------
        state: tuple
            A state in the game.

        Returns
        -------
        prior_probs: dict
            A dictionary from actions to probabilities. Some actions might not be
            legal in this game state, but the evaluator returns a probability for
            choosing each one.
        value: float
            The evaluator's estimate of the value of the state 'state'.
        """
        next_states = game.legal_actions(state)
        uniform_prior_probs = {action: 1 / len(next_states) for action in next_states}
        return uniform_prior_probs, 0

    return trivial_estimator


def create_rollout_estimator(game, num_rollouts):
    # TODO: test this and write docstring
    def rollout_estimator(state):
        next_states = game.legal_actions(state)
        uniform_prior_probs = {action: 1 / len(next_states) for action in next_states}
        player_no = game.current_player(state)
        # Snapshot the root state and roll out from a fresh copy each
        # iteration; reassigning `state` in the loop previously left
        # it terminal so every rollout after the first was a no-op.
        root_state = state
        total_value = 0
        for _ in range(num_rollouts):
            s = root_state
            while not game.is_terminal(s):
                next_states = game.legal_actions(s)
                s = random.choice(list(next_states.values()))

            total_value += game.utility(s)[player_no]
        mean_value = total_value / num_rollouts

        return uniform_prior_probs, mean_value

    return rollout_estimator


@dataclass
class NetConfig:
    """Per-game configuration for :class:`PolicyValueNet`.

    Attributes
    ----------
    board_h, board_w:
        Spatial dimensions of the board (height, width) for the ``(N, C, H, W)``
        conv input.
    in_channels:
        Number of input channels (e.g. 1 for a single board plane, 2 for the two
        NAC 3x6 bitboard planes).
    conv_specs:
        ``(out_channels, kernel_size)`` for each ``Conv2d`` layer in order.
    trunk_dims:
        Hidden dims of the shared dense trunk (each followed by ReLU).
    value_head_dims, policy_head_dims:
        Hidden dims of the optional value/policy head towers. Empty towers read
        straight off the trunk.
    pi_dim:
        Number of policy logits (the action space size).
    """

    board_h: int
    board_w: int
    in_channels: int
    conv_specs: list[tuple[int, int]]
    trunk_dims: list[int]
    pi_dim: int
    value_head_dims: list[int] = field(default_factory=list)
    policy_head_dims: list[int] = field(default_factory=list)


class PolicyValueNet(nn.Module):
    """A single configurable policy/value CNN.

    Reproduces the three pinned TF1 topologies via a :class:`NetConfig`. All
    convolutions are ``Conv2d`` with ``padding="same"`` (fixing the old
    3D-conv bug) and stride 1, so spatial dims are preserved and the flattened
    trunk input is ``out_channels * board_h * board_w``. ``forward`` returns
    ``(policy_logits, value)`` in a single pass; the value head uses a
    ``tanh`` activation and the policy head has no activation.
    """

    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        self.board_h = cfg.board_h
        self.board_w = cfg.board_w

        convs: list[nn.Module] = []
        channels = cfg.in_channels
        for out_ch, kernel in cfg.conv_specs:
            convs.append(
                nn.Conv2d(channels, out_ch, kernel_size=kernel, padding="same")
            )
            convs.append(nn.ReLU())
            channels = out_ch
        self.conv = nn.Sequential(*convs)

        # "same" padding preserves H and W, so the flattened size is exact.
        flat = channels * cfg.board_h * cfg.board_w
        self.trunk = self._mlp(flat, cfg.trunk_dims)
        trunk_out = cfg.trunk_dims[-1] if cfg.trunk_dims else flat

        self.value_tower = self._mlp(trunk_out, cfg.value_head_dims)
        self.policy_tower = self._mlp(trunk_out, cfg.policy_head_dims)
        v_in = cfg.value_head_dims[-1] if cfg.value_head_dims else trunk_out
        p_in = cfg.policy_head_dims[-1] if cfg.policy_head_dims else trunk_out
        self.value_out = nn.Linear(v_in, 1)
        self.policy_out = nn.Linear(p_in, cfg.pi_dim)

    @staticmethod
    def _mlp(in_dim: int, dims: list[int]) -> nn.Sequential:
        """Build a ReLU MLP. An empty ``dims`` yields an identity passthrough."""
        layers: list[nn.Module] = []
        d = in_dim
        for out in dims:
            layers.append(nn.Linear(d, out))
            layers.append(nn.ReLU())
            d = out
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Verify Conv2d runs on the true (H, W) board so a
        # transpose of a rectangular board (e.g. NAC 3x6 -> 6x3) fails loudly.
        if tuple(x.shape[-2:]) != (self.board_h, self.board_w):
            raise ValueError(
                f"expected input (H, W) = ({self.board_h}, {self.board_w}), "
                f"got {tuple(x.shape[-2:])}"
            )
        z = self.conv(x).flatten(1)
        z = self.trunk(z)
        value = torch.tanh(self.value_out(self.value_tower(z)))
        policy_logits = self.policy_out(self.policy_tower(z))
        return policy_logits, value


def _iter_batches(data, batch_size):
    """Yield consecutive NON-overlapping batches of ``batch_size``.

    The TF1 ``loss`` used a sliding window ``range(i, i + batch_size)`` which
    overlapped batches; this steps by ``batch_size`` so each row is used at most
    once.
    """
    for start in range(0, len(data) - batch_size + 1, batch_size):
        yield data[start : start + batch_size]


class AbstractNeuralNetEstimator(abc.ABC):
    game_state_shape = NotImplemented
    action_indices = NotImplemented

    def __init__(
        self,
        learning_rate=1e-2,
        l2_weight=1e-4,
        value_weight=1,
        device: str | None = None,
    ):
        self.learning_rate = learning_rate
        self.l2_weight = l2_weight
        self.value_weight = value_weight
        # Additive device= (default None -> CPU). Stored for the eager
        # resolve in _initialise_net so the zero-arg create_estimator() factory
        # and positional constructions stay valid.
        self._device_arg = device
        self._initialise_net()

    @abc.abstractmethod
    def _config(self) -> NetConfig:
        """Return the per-game network configuration."""

    @abc.abstractmethod
    def _state_to_vector(self, state):
        """Map the state to a flat ``(N, in_channels * board_h * board_w)``
        numpy array suitable for input to the network."""

    def _initialise_net(self):
        """Build the network, optimizer and step counter (replaces the TF
        graph/session)."""
        # Resolve the compute device once, then place the net on it
        # BEFORE constructing the optimizer (its parameters must already be
        # on-device).
        self.device = get_device(self._device_arg)
        self.cfg = self._config()
        self.net = PolicyValueNet(self.cfg).to(self.device)
        # Direct equivalent of the old MomentumOptimizer(lr, momentum=0.9);
        # SGD is the real optimizer, NOT AdamW.
        self.optimizer = torch.optim.SGD(
            self.net.parameters(), lr=self.learning_rate, momentum=0.9
        )
        # Global_step stays a plain int.
        self.global_step = 0

    def _vectors_to_input(self, vectors) -> torch.Tensor:
        """Reshape a flat ``(N, flat)`` numpy array to an ``(N, C, H, W)`` tensor.

        The ``float32`` cast is the load-bearing numpy->tensor boundary defense
        (float64 crashes on MPS); the reshaped tensor is then
        moved to the resolved ``self.device``.
        """
        x = torch.as_tensor(np.asarray(vectors), dtype=torch.float32)
        x = x.reshape(-1, self.cfg.in_channels, self.cfg.board_h, self.cfg.board_w)
        return x.to(self.device)

    def _batch_to_tensors(self, batch):
        """Encode a list of ``(state, pi, z)`` rows into ``(x, pi, z)`` tensors."""
        vectors = np.concatenate([self._state_to_vector(x[0]) for x in batch], axis=0)
        pis = np.array([x[1] for x in batch], dtype=np.float32)
        zs = np.array([x[2] for x in batch], dtype=np.float32).reshape(-1, 1)
        # x is already on self.device via _vectors_to_input; move pi/z too.
        x = self._vectors_to_input(vectors)
        pi = torch.as_tensor(pis, dtype=torch.float32).to(self.device)
        z = torch.as_tensor(zs, dtype=torch.float32).to(self.device)
        return x, pi, z

    def __call__(self, state):
        """Returns the result of the neural net applied to the state. This is
        'probs' and 'value'.

        Parameters
        ----------
        state: ndarray
            The input state to the network. We expect a single state; if a batch
            is passed the first state's outputs are returned (consistent with
            ``probs`` below slicing the first row via ``action_indices``).

        Returns
        -------
        probs: dict
            The probabilities returned by the net as a dictionary keyed by ALL
            actions in ``action_indices`` (unmasked; the legal/sum-to-1
            invariant is realized at the MCTS boundary).
        value: float
            The value returned by the net.
        """
        vectors = self._state_to_vector(state)
        x = self._vectors_to_input(vectors)

        self.net.eval()
        with torch.no_grad():
            logits, value = self.net(x)
            probs = F.softmax(logits, dim=1)

        # Numpy only accepts CPU tensors; .cpu() is a no-op on CPU
        # and the required marshal off MPS.
        probs = probs.cpu().numpy().ravel()
        value = value.cpu().numpy().ravel()[0]

        probs_dict = {
            action: probs[index] for action, index in self.action_indices.items()
        }

        return probs_dict, float(value)

    def _compute_loss(self, x, pi, z):
        """Compute the combined loss and its value/probs components.

        Soft-target policy cross-entropy via ``F.log_softmax`` (a hand-rolled
        soft-CE -- the policy targets are soft distributions, not class
        indices), MSE value loss, and an EXPLICIT L2 penalty. NOTE: the
        TF1 code never added L2 to
        the minimized loss (old ``estimator.py:465``), so this explicit L2 is a
        documented behaviour change. ``.mean()`` (not ``.sum(dim=1).mean()``)
        matches the TF1 ``reduce_mean`` scale.
        """
        logits, value = self.net(x)
        log_p = F.log_softmax(logits, dim=1)
        loss_probs = -(pi * log_p).mean()
        loss_value = F.mse_loss(value, z)
        l2 = sum(
            p.pow(2).sum() for n, p in self.net.named_parameters() if "weight" in n
        )
        loss = self.value_weight * loss_value + loss_probs + self.l2_weight * l2
        return loss, loss_value, loss_probs

    def loss(self, data, batch_size):
        """Computes the loss of the network on the data.

        Parameters
        ----------
        data: list
            A list consisting of (state, probs, z) tuples, where player is the
            player in the state and z is the utility to player in the last state
            from the corresponding self-play game.
        batch_size: int

        Returns
        -------
        loss: float
            The loss of the network on the given data.
        loss_value: float
            The loss of the value part of the network.
        loss_probs: float
            The loss of the probability part of the network.
        """
        losses = []
        loss_value_list = []
        loss_probs_list = []

        self.net.eval()
        with torch.no_grad():
            for batch in _iter_batches(data, batch_size):
                x, pi, z = self._batch_to_tensors(batch)
                loss, loss_value, loss_probs = self._compute_loss(x, pi, z)
                losses.append(loss.item())
                loss_value_list.append(loss_value.item())
                loss_probs_list.append(loss_probs.item())

            # A dataset smaller than `batch_size` yields no full batch;
            # fall back to a single ragged batch over all the data so
            # the loss is a real number rather than a silent NaN from
            # ``np.mean([])``.
            if not losses:
                x, pi, z = self._batch_to_tensors(data)
                loss, loss_value, loss_probs = self._compute_loss(x, pi, z)
                return loss.item(), loss_value.item(), loss_probs.item()

        return np.mean(losses), np.mean(loss_value_list), np.mean(loss_probs_list)

    def train_step(self, batch, return_summary=False):
        """Trains the network on the batch.

        Parameters
        ----------
        batch: list
            A list consisting of (state, probs, z) tuples, where player
            is the player in the state and z is the utility to player in
            the last state from the corresponding self-play game.
        return_summary: bool
            Whether to return the scalar loss (replaces the old TF
            summary tensor).

        Returns
        -------
        loss: float or None
            The scalar loss on the batch when ``return_summary`` is True.
        """
        self.net.train()
        x, pi, z = self._batch_to_tensors(batch)
        loss, _, _ = self._compute_loss(x, pi, z)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the global step
        self.global_step += 1
        if return_summary:
            return float(loss.item())

    def train(
        self,
        training_data,
        batch_size,
        training_iters,
        mode="reinforcement",
        writer=None,
        verbose=True,
    ):
        """Trains the net on the training data.

        Parameters
        ----------
        training_data: list
            A list consisting of (state, probs, z) tuples, where player
            is the player in the state and z is the utility to player in
            the last state from the corresponding self-play game.
        batch_size: int
        training_iters: int
            The number of training iterations to run, where a training
            iteration corresponds to updating the net on a single batch
            of training data. If this is set to -1 in supervised mode,
            it will run for a whole epoch, i.e. process the entire data
            set exactly once.
        mode: str, {'reinforcement', 'supervised'}
            The mode of training. If running in reinforcement mode then
            the batch data are sampled randomly from the training data
            at each training iteration. If running in supervised mode,
            then the data is randomly ordered and then each training
            iteration steps through the data in batches.
        writer:
            Retained for signature compatibility; summary logging is currently
            a no-op.
        verbose: bool
            Print out progress if True, else don't print anything.
        """
        # TODO: This concrete implementation of two cases probably shouldn't be in ABC

        if mode not in ["reinforcement", "supervised"]:
            raise ValueError("`mode` must be 'reinforcement', 'supervised'.")

        if mode == "reinforcement":
            if training_iters == -1:
                raise ValueError("`training_iters` must be > 1 for reinforcement mode.")
            self._train_reinforcement(
                training_data, batch_size, training_iters, writer, verbose
            )
        elif mode == "supervised":
            self._train_supervised(
                training_data, batch_size, training_iters, writer, verbose
            )

    def _train_reinforcement(
        self, training_data, batch_size, training_iters, writer, verbose
    ):
        """Train the net in reinforcement learning mode.

        In this case, a random batch is sampled for the data every
        training iteration. This may mean that the same data points are
        trained on multiple times before the every data point is in the
        training data is considered.
        """
        disable_tqdm = not verbose
        for _ in tqdm(range(training_iters), disable=disable_tqdm):
            batch_indices = np.random.choice(len(training_data), batch_size)
            batch = [training_data[ix] for ix in batch_indices]
            self.train_step(batch, return_summary=True)

    def _train_supervised(
        self, training_data, batch_size, training_iters, writer, verbose
    ):
        """Train the net in supervised learning mode.

        In this case, the training data are randomly shuffled and then
        they are processed sequentially in batches. The number of
        batches trained on is equal to the number training iterations.
        """
        size = len(training_data)
        training_indices = [i for i in range(size)]
        random.shuffle(training_indices)

        # calculate training iterations for single epoch if required
        if training_iters == -1:
            training_iters = (len(training_data) + batch_size - 1) // batch_size

        # generate batch indices, the final batch may be smaller if
        # `batch_size` doesn't evenly divide into size of training data
        batch_indices_list = [
            training_indices[i * batch_size : min(size, (i + 1) * batch_size)]
            for i in range(training_iters)
        ]
        # Drop empty trailing slices when `training_iters` exceeds the batch
        # count; otherwise `train_step([])` hits `np.concatenate([])`.
        batch_indices_list = [b for b in batch_indices_list if b]

        disable_tqdm = not verbose
        for batch_indices in tqdm(batch_indices_list, disable=disable_tqdm):
            batch = [training_data[ix] for ix in batch_indices]
            self.train_step(batch, return_summary=True)

    def create_estimate_fn(self):
        """Returns an evaluator function corresponding to the neural network.

        Note that we expect self.action_indices to be a dictionary with keys
        the available actions and values the index of that action. Indices
        must be unique in 0, 1, .., #actions-1.

        Returns
        -------
        estimate: func
            A function that evaluates states.
        """

        return self.__call__

    def save(self, save_file):
        """Saves the net to ``save_file`` as a ``.pt`` bundle.

        Bundles the model weights, optimizer state and ``global_step`` so a
        training run can resume.
        """
        torch.save(
            {
                "model": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "global_step": self.global_step,
            },
            save_file,
        )

    def restore(self, save_file):
        """Restore the net from ``save_file``.

        ``map_location=self.device`` remaps the checkpoint onto the estimator's
        resolved device; ``self.net.to(self.device)`` after the load is
        belt-and-braces for cross-device restores.
        """
        checkpoint = torch.load(save_file, map_location=self.device, weights_only=True)
        self.net.load_state_dict(checkpoint["model"])
        self.net.to(self.device)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.global_step = checkpoint["global_step"]


class NACNetEstimator(AbstractNeuralNetEstimator):
    game_state_shape = (1, 9)

    def __init__(
        self,
        learning_rate=1e-2,
        l2_weight=1e-4,
        action_indices=None,
        value_weight=1,
        device: str | None = None,
    ):
        super().__init__(learning_rate, l2_weight, value_weight, device)
        self.action_indices = action_indices

    def _config(self) -> NetConfig:
        # NAC 3x3: in (N, 1, 3, 3); conv 8->16->16 k=2; trunk dense 32;
        # value dense 1 (tanh); policy dense 9. BN was inactive here (no-op
        # removal). The old third conv layer was a 3D-conv BUG -> Conv2d.
        return NetConfig(
            board_h=3,
            board_w=3,
            in_channels=1,
            conv_specs=[(8, 2), (16, 2), (16, 2)],
            trunk_dims=[32],
            pi_dim=9,
        )

    def _state_to_vector(self, state):
        state = np.array(state).reshape((-1, 9))
        return np.nan_to_num(state)


class NAC3x6NetEstimator(AbstractNeuralNetEstimator):
    game_state_shape = (1, 36)

    def __init__(
        self,
        learning_rate=1e-2,
        l2_weight=1e-4,
        action_indices=None,
        value_weight=1,
        device: str | None = None,
    ):
        super().__init__(learning_rate, l2_weight, value_weight, device)
        self.action_indices = action_indices

    @staticmethod
    def _binary_state_to_array(state):
        player1_board = [int(i) for i in f"{state[0]:018b}"]
        player2_board = [int(i) for i in f"{state[1]:018b}"]
        return player1_board + player2_board

    def _config(self) -> NetConfig:
        # NAC 3x6: in (N, 2, 3, 6) -- the rectangular-board net, so a
        # (N, 2, 6, 3) transpose fails loudly in PolicyValueNet.forward.
        # conv 32->64->128->128 k=2 (old third/fourth conv layers were 3D-conv
        # BUGs -> Conv2d); trunk 128->128->256; value tower 128->64;
        # policy tower
        # 256->128; pi_dim 18.
        # NOTE: the TF1 net had use_batch_norm=True ACTIVE
        # (old estimator.py:689). Dropping BatchNorm here is a REAL behaviour
        # change for NAC3x6 (not a no-op as it is for NAC3x3 / Connect Four).
        return NetConfig(
            board_h=3,
            board_w=6,
            in_channels=2,
            conv_specs=[(32, 2), (64, 2), (128, 2), (128, 2)],
            trunk_dims=[128, 128, 256],
            value_head_dims=[128, 64],
            policy_head_dims=[256, 128],
            pi_dim=18,
        )

    def _state_to_vector(self, state):
        # Give NAC3x6 a real
        # _state_to_vector that wraps the binary encoder so the SINGLE base
        # __call__ and create_estimate_fn work uniformly for all three nets
        # (removing the old custom estimate_fn override) without changing
        # externally observed behaviour.
        return np.array(self._binary_state_to_array(state)).reshape((-1, 36))


class ConnectFourNet(AbstractNeuralNetEstimator):
    game_state_shape = (1, 42)

    def __init__(
        self,
        learning_rate=1e-2,
        l2_weight=1e-4,
        action_indices=None,
        value_weight=1,
        device: str | None = None,
    ):
        super().__init__(learning_rate, l2_weight, value_weight, device)
        self.action_indices = action_indices

    def _config(self) -> NetConfig:
        # Connect Four: in (N, 1, 6, 7); conv 8->16->32->64 k=3 (all conv2d in
        # the TF1 code -- no bug); trunk 64->128->256; value dense 1 (tanh);
        # policy dense 7. No BatchNorm.
        return NetConfig(
            board_h=6,
            board_w=7,
            in_channels=1,
            conv_specs=[(8, 3), (16, 3), (32, 3), (64, 3)],
            trunk_dims=[64, 128, 256],
            pi_dim=7,
        )

    def _state_to_vector(self, state):
        # Preserve the asymmetry vs NAC: NO nan_to_num here (the TF1 code did not
        # apply it for Connect Four).
        return np.array(state).reshape((-1, 42))
