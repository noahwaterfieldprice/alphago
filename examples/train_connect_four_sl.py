"""This file trains a connect four net with supervised learning."""

import argparse
import os
import pickle
import time
from collections import deque

import numpy as np

from alphago.alphago import optimise_estimator
from alphago.connect_four_data import load_solved_states
from alphago.estimator import (
    ConnectFourNet,
    create_rollout_estimator,
    create_trivial_estimator,
)
from alphago.evaluator import run_gauntlet
from alphago.games.connect_four import ConnectFour
from alphago.player import MCTSPlayer, RandomPlayer


def compute_checkpoint_name(step, path):
    return path + f"{step}.pt"


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


def train_network(training_data, evaluate_every):
    np.random.shuffle(training_data)
    dev_fraction = 0.02
    num_dev = int(dev_fraction * len(training_data))
    dev_data = training_data[:num_dev]
    training_data = training_data[num_dev:]

    # Comparison players for evaluation
    mcts_iters = 10
    game = ConnectFour()
    trivial_estimator = create_trivial_estimator(game)
    rollout_estimator = create_rollout_estimator(game, 50)
    random_player = RandomPlayer(game)
    c_puct = 0.5
    MCTSPlayer(game, trivial_estimator, mcts_iters, c_puct, 0.01)
    MCTSPlayer(game, rollout_estimator, mcts_iters, c_puct, 0.01)
    # fixed_comparison_players = {1: random_player,
    #                             2: trivial_mcts_player,
    #                             3: rollout_mcts_player}

    fixed_comparison_players = {1: random_player}

    supervised_player_no = len(fixed_comparison_players) + 1
    supervised_players_queue = deque(maxlen=2)

    # Hyperparameters
    learning_rate = 1e-4
    batch_size = 32
    l2_weight = 1e-1
    value_weight = 1e-2
    num_train = len(training_data)

    checkpoint_every = evaluate_every
    num_steps = 1000

    # Build the hyperparameter string
    hyp_string = (
        f"lr={learning_rate},batch_size={batch_size},"
        f"value_weight={value_weight},l2_weight={l2_weight},num_train={num_train}"
    )

    game_name = "connect_four-sl"

    current_time_format = time.strftime("%Y-%m-%d_%H:%M:%S")
    path = f"experiments/{game_name}-{hyp_string}-{current_time_format}/"
    checkpoint_path = path + "checkpoints/"
    game_results_file_name = path + "game_results.pickle"

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
            writer=None,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("training_data", help="Input file with training data.")
    parser.add_argument(
        "--max_lines",
        help="The maximum number of lines to read in from the training data.",
    )
    parser.add_argument(
        "--evaluate_every", help="The number of epochs between evaluatingiterations."
    )
    parser.add_argument(
        "--evaluate_checkpoint_path",
        help="The checkpoint path to evaluate. If given, "
        "then evaluate_step must also be provided.",
    )
    parser.add_argument(
        "--evaluate_step", help="The step of the checkpoint to evaluate."
    )

    args = parser.parse_args()

    # Load the training data as (state, probs_vector, z) tuples.
    max_lines = int(args.max_lines) if args.max_lines is not None else None

    training_data = load_solved_states(args.training_data, max_lines=max_lines)

    # If evaluate checkpoint path is given, then just evaluate that network.
    if args.evaluate_checkpoint_path is not None:
        checkpoint_path = args.evaluate_checkpoint_path
        checkpoint_step = args.evaluate_step

        estimator = load_net(checkpoint_step, checkpoint_path)

        optimal_actions_list = [
            (state, probs_vector_to_optimal_actions(probs_vector))
            for state, probs_vector, _ in training_data
        ]

        accuracy = compute_accuracy(estimator, optimal_actions_list)
        print(f"Accuracy: {accuracy}")
    else:
        # Otherwise, train the network.
        evaluate_every = 5
        if args.evaluate_every is not None:
            evaluate_every = int(args.evaluate_every)

        train_network(training_data, evaluate_every)
