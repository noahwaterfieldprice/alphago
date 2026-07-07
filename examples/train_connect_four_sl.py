"""This file trains a connect four net with supervised learning."""

import os
import pickle
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from omegaconf import MISSING

from alphago.alphago import compute_checkpoint_name, optimise_estimator
from alphago.config import load_config, resolve_paths
from alphago.connect_four_data import load_solved_states
from alphago.estimator import (
    ConnectFourNet,
    create_rollout_estimator,
    create_trivial_estimator,
)
from alphago.evaluator import run_gauntlet
from alphago.games.connect_four import ConnectFour
from alphago.player import MCTSPlayer, RandomPlayer


@dataclass
class SupervisedPathsConfig:
    """Filesystem paths for the supervised experiment.

    Attributes:
        training_data: Input solver-data file (``MISSING`` so an omitted path
            fails loud at startup). Invoke as
            ``paths.training_data=data/connect_four_data.txt``.
        experiment_dir: Absolute run directory; derived by :func:`resolve_paths`.
        checkpoint_dir: Absolute checkpoint directory; derived likewise.
    """

    training_data: str = MISSING
    experiment_dir: str | None = None
    checkpoint_dir: str | None = None


@dataclass
class SupervisedConfig:
    """Supervised Connect Four training configuration.

    Attributes:
        paths: Input-data and experiment paths.
        max_lines: Maximum lines to read from the training data (``None`` = all).
        evaluate_every: Steps between checkpoint + gauntlet evaluations.
        evaluate_checkpoint_path: If set, evaluate this checkpoint instead of
            training (``evaluate_step`` must also be provided).
        evaluate_step: The checkpoint step to evaluate.
        learning_rate: Optimizer learning rate.
        l2_weight: L2 regularization weight.
        value_weight: Relative weight of the value loss term.
        batch_size: Mini-batch size.
        num_steps: Number of supervised training steps.
        mcts_iters: Simulations per move for the evaluation players.
        c_puct: PUCT exploration constant for the evaluation players.
    """

    paths: SupervisedPathsConfig = field(default_factory=SupervisedPathsConfig)
    max_lines: int | None = None
    evaluate_every: int = 5
    evaluate_checkpoint_path: str | None = None
    evaluate_step: int | None = None
    learning_rate: float = 1e-4
    l2_weight: float = 1e-1
    value_weight: float = 1e-2
    batch_size: int = 32
    num_steps: int = 1000
    mcts_iters: int = 10
    c_puct: float = 0.5


def probs_vector_to_optimal_actions(probs_vector):
    """Recovers the 1-indexed optimal actions from a solver probs vector.

    The solver-data loader (``alphago.connect_four_data.load_solved_states``)
    encodes the optimal actions as a probability vector that is uniform over the
    optimal columns and zero elsewhere. The optimal actions are therefore the
    (1-indexed) columns with positive probability.

    Args:
        probs_vector: A length-7 one-hot/uniform policy target over the columns.

    Returns:
        A list of the optimal actions, indexed 1 to 7.
    """
    return [i + 1 for i, prob in enumerate(probs_vector) if prob > 0]


def update_results(game_results, game_results_file_name):
    """Update the results stored in the pickle file game_results_file_name.
    Loads the game results from the file (if it exists), then adds
    game_results to the results, and saves to game_results_file_name.

    The file game_results_file_name stores a pickle encoding of a dictionary
    with keys (i, j) pairs and values n, denoting that i beat j n times.

    Parameters
    ----------
    game_results: list
        A list of (i, j, n) tuples, meaning player i scored n against player j.
    game_results_file_name: str
        The path to save the updated results to. Creates this path if it
        doesn't exist.
    """
    print("Game results", game_results)
    # Load results from file
    if os.path.exists(game_results_file_name):
        with open(game_results_file_name, "rb") as f:
            results = pickle.load(f)
    else:
        results = {}

    # Update the results dictionary with the game results.
    for result in game_results:
        ij = result[:2]
        score = result[2]
        if ij not in results:
            results[ij] = score
        else:
            results[ij] += score

    print("Results", results)

    with open(game_results_file_name, "wb") as f:
        pickle.dump(results, f)


def compute_accuracy(estimator, optimal_actions):
    """Computes the accuracy of the estimator predicting actions according to
    the maximum probability.

    Parameters
    ----------
    estimator: func or AbstractNeuralNetEstimator
        Can be evaluated on a state to get a value and probabilities over
        actions.
    optimal_actions: list
        A list of tuples. Each tuple is of the form (state,
        optimal_actions). Here state is a connect four state, and
        optimal_actions is a list of optimal actions in that state. The
        actions are all indexed 1 to 7.

    Returns
    -------
    float
        The accuracy of the network at predicting optimal actions. This is the
        fraction of solved_states for which the network's maximal
        probability action is in the optimal actions.
    """
    predicted_actions = []
    actions_list = []
    for state, actions in optimal_actions:
        probs, _ = estimator(state)

        # Get the estimator's predicted action in the range 1 up to 7.
        predicted_action = max(probs, key=probs.get) + 1

        predicted_actions.append(predicted_action)
        actions_list.append(actions)

    return np.mean(
        [
            1 if predicted_actions[i] in actions_list[i] else 0
            for i in range(len(predicted_actions))
        ]
    )


# Training data is loaded via `alphago.connect_four_data.load_solved_states`,
# which parses the c4solver output (one record per line in the format
# `<moves> <opt_action> <value>`) into (state, probs_vector, z) training tuples.


def load_net(step, checkpoint_path):
    """Evaluates the network saved in the checkpoint path for the given step."""
    game = ConnectFour()
    estimator = ConnectFourNet(
        learning_rate=1e-4,
        l2_weight=1e-4,
        value_weight=0.01,
        action_indices=game.action_indices,
    )
    checkpoint_name = compute_checkpoint_name(step, checkpoint_path)
    estimator.restore(checkpoint_name)
    return estimator


def train_network(cfg, training_data):
    """Trains a Connect Four net by supervised learning from solver data.

    Args:
        cfg: The resolved :class:`SupervisedConfig`.
        training_data: A list of ``(state, probs_vector, z)`` training tuples.
    """
    np.random.shuffle(training_data)
    dev_fraction = 0.02
    # max(1, ...) so a small dataset (<50 rows, where int(0.02 * len) rounds to
    # 0) never collapses to an empty dev split; compute_accuracy
    # and estimator.loss then always have at least one held-out row.
    num_dev = max(1, int(dev_fraction * len(training_data)))
    dev_data = training_data[:num_dev]
    training_data = training_data[num_dev:]

    # Comparison players for evaluation
    mcts_iters = cfg.mcts_iters
    game = ConnectFour()
    trivial_estimator = create_trivial_estimator(game)
    rollout_estimator = create_rollout_estimator(game, 50)
    random_player = RandomPlayer(game)
    c_puct = cfg.c_puct
    MCTSPlayer(game, trivial_estimator, mcts_iters, c_puct, 0.01)
    MCTSPlayer(game, rollout_estimator, mcts_iters, c_puct, 0.01)
    # fixed_comparison_players = {1: random_player,
    #                             2: trivial_mcts_player,
    #                             3: rollout_mcts_player}

    fixed_comparison_players = {1: random_player}

    supervised_player_no = len(fixed_comparison_players) + 1
    supervised_players_queue = deque(maxlen=2)

    # Hyperparameters
    learning_rate = cfg.learning_rate
    batch_size = cfg.batch_size
    l2_weight = cfg.l2_weight
    value_weight = cfg.value_weight

    checkpoint_every = cfg.evaluate_every
    num_steps = cfg.num_steps

    # Derive absolute experiment/checkpoint dirs in place when unset.
    resolve_paths(cfg, "connect_four-sl")
    checkpoint_path = cfg.paths.checkpoint_dir
    game_results_file_name = str(Path(cfg.paths.experiment_dir) / "game_results.pickle")

    # Create the experiment/checkpoint directories up front; the TF
    # `Saver` used to create them implicitly, but `torch.save` does not, so the
    # first checkpoint would otherwise raise FileNotFoundError.
    os.makedirs(checkpoint_path, exist_ok=True)

    estimator = ConnectFourNet(
        learning_rate=learning_rate,
        l2_weight=l2_weight,
        value_weight=value_weight,
        action_indices=game.action_indices,
    )

    verbose = True
    training_iters = -1

    # TensorBoard summary logging was removed with the TF port; dev metrics
    # are printed below.
    dev_optimal_actions = [
        (state, probs_vector_to_optimal_actions(probs_vector))
        for state, probs_vector, value in dev_data
    ]

    for step in range(num_steps):
        print(f"Step: {step}")
        optimise_estimator(
            estimator,
            training_data,
            batch_size,
            training_iters,
            mode="supervised",
            verbose=verbose,
        )

        # Now compute dev loss
        dev_loss, dev_loss_value, dev_loss_probs = estimator.loss(dev_data, batch_size)
        dev_accuracy = compute_accuracy(estimator, dev_optimal_actions)
        print(
            f"Dev loss: {dev_loss}, dev loss value: {dev_loss_value}, "
            f"dev loss probs: {dev_loss_probs}, dev accuracy: {dev_accuracy}"
        )

        if step % checkpoint_every == 0 and step > 0:
            checkpoint_name = compute_checkpoint_name(step, checkpoint_path)
            estimator.save(checkpoint_name)

            new_estimator = ConnectFourNet(
                learning_rate=learning_rate,
                l2_weight=l2_weight,
                value_weight=value_weight,
                action_indices=game.action_indices,
            )
            new_estimator.restore(checkpoint_name)

            new_player = MCTSPlayer(game, new_estimator, mcts_iters, c_puct)

            supervised_players = {j: player for j, player in supervised_players_queue}
            comparison_players = {**fixed_comparison_players, **supervised_players}

            game_results = run_gauntlet(
                game, (supervised_player_no, new_player), comparison_players, 1
            )

            update_results(game_results, game_results_file_name)

            supervised_players_queue.appendleft((supervised_player_no, new_player))
            supervised_player_no += 1


def main(cfg) -> None:
    """Train, or evaluate a checkpoint, per the resolved config.

    Args:
        cfg: The resolved :class:`SupervisedConfig`.
    """
    # Load the training data as (state, probs_vector, z) tuples. Accessing the
    # MISSING training_data path here fails loud at startup if it was omitted.
    training_data = load_solved_states(cfg.paths.training_data, max_lines=cfg.max_lines)

    # If an evaluate checkpoint path is given, then just evaluate that network.
    if cfg.evaluate_checkpoint_path is not None:
        # An evaluate path without evaluate_step cannot resolve which
        # checkpoint to load; fail loud (what + why + fix) rather than passing
        # evaluate_step=None into compute_checkpoint_name and crashing opaquely.
        if cfg.evaluate_step is None:
            raise RuntimeError(
                "evaluate_checkpoint_path is set but evaluate_step is None: "
                "cannot resolve which checkpoint step to evaluate. Set "
                "evaluate_step=<int> to the checkpoint step to evaluate, or "
                "unset evaluate_checkpoint_path to train instead."
            )
        estimator = load_net(cfg.evaluate_step, cfg.evaluate_checkpoint_path)

        optimal_actions_list = [
            (state, probs_vector_to_optimal_actions(probs_vector))
            for state, probs_vector, _ in training_data
        ]

        accuracy = compute_accuracy(estimator, optimal_actions_list)
        print(f"Accuracy: {accuracy}")
    else:
        # Otherwise, train the network.
        train_network(cfg, training_data)


if __name__ == "__main__":
    main(load_config(SupervisedConfig))
