import numpy as np
import torch
import torch.nn as nn


class MockNetEstimator(nn.Module):
    """A tiny torch policy/value net used as an estimator test fixture.

    Satisfies the frozen estimator contract ``(state) -> (probs_dict, value)``
    so it can be driven through ``mcts``. ``probs_dict`` is keyed by all
    ``action_indices`` (mirroring the real estimators); a raw probs array would
    break ``normalise_distribution`` in ``mcts_tree`` (which expects a dict).
    """

    def __init__(self, learning_rate, action_indices=None):
        super().__init__()
        if action_indices is None:
            action_indices = {i: i for i in range(18)}
        self.action_indices = action_indices
        self.dense1 = nn.Linear(1, 20)
        self.dense2 = nn.Linear(20, 20)
        self.value_head = nn.Linear(20, 1)
        self.policy_head = nn.Linear(20, 18)

    def forward(self, x):
        z = torch.relu(self.dense1(x))
        z = torch.relu(self.dense2(z))
        value = torch.tanh(self.value_head(z))
        prob_logits = self.policy_head(z)
        return prob_logits, value

    def __call__(self, state):
        """Returns the result of the neural net applied to the state, as a
        ``(probs_dict, value)`` pair.

        Returns
        -------
        probs: dict
            The probabilities returned by the net keyed by all actions.
        value: float
            The value returned by the net.
        """
        if not hasattr(state, "__len__"):
            state = (state,)

        x = torch.as_tensor(np.asarray(state), dtype=torch.float32).reshape(-1, 1)
        with torch.no_grad():
            prob_logits, value = self.forward(x)
            probs = torch.softmax(prob_logits, dim=1)

        probs = probs.numpy().ravel()
        [value] = value.numpy().ravel()

        probs_dict = {
            action: probs[index] for action, index in self.action_indices.items()
        }

        return probs_dict, float(value)
