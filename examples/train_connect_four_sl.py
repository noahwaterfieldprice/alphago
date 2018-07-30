"""This file trains a connect four net with supervised learning.
"""
import argparse
import collections
import pickle
import os
import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from alphago.games.connect_four import action_list_to_state, ConnectFour
from alphago.estimator import (ConnectFourNet, create_trivial_estimator,
                               create_rollout_estimator)
from alphago.evaluator import run_gauntlet
from alphago.alphago import optimise_estimator
from tools.summary_scalars import SummaryScalars
from alphago.player import RandomPlayer, MCTSPlayer


def compute_checkpoint_name(step, path):
    return path + "{}.checkpoint".format(step)


def solved_states_to_training_data(solved_states):
    """Converts a list of tuples to training data for AlphaGo.

    Parameters
    ----------
    solved_states: list
        A list, with each element being a tuple. The tuples are of the form
        (action_list, action, outcome), where action_list is the list of played
        columns, indexed from 1 to 7; action is the optimal action to play in
        the state, and outcome is the eventual outcome to the current player.

    Returns
    -------
    training_data: list
        A list of training data suitable for AlphaGo.
    """
    training_data = []
    for action_list, action, outcome in solved_states:
        # Convert the action list to a state for connect four.
        state = action_list_to_state([a - 1 for a in action_list])

        # Set the probs vector to be 1 for the optimal action, and 0 for all
        # other actions.
        probs_vector = np.zeros(7, float)
        probs_vector[action - 1] = 1

        # Store in training_data.
        training_data.append((state, probs_vector, outcome))

    return training_data


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
        with open(game_results_file_name, 'rb') as f:
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

    with open(game_results_file_name, 'wb') as f:
        pickle.dump(results, f)


def compute_accuracy(estimator, solved_states):
    """Computes the accuracy of the estimator predicting moves according to
    the maximum probability.
    """
    actions = []
    predicted_actions = []
    print("Computing accuracy")
    for action_list, action, _ in tqdm(solved_states):
        state = action_list_to_state([a - 1 for a in action_list])
        probs, val = estimator(state)
        predicted_action = max(probs, key=probs.get) + 1

        predicted_actions.append(predicted_action)
        actions.append(action)

    return np.mean([1 if actions[i] == predicted_actions[i] else 0 for i in
                    range(len(actions))])


# Training data should be a file with lines of the form:
# action_list action outcome
# This is as output by c4solver.

def load_net(step, checkpoint_path):
    """Evaluates the network saved in the checkpoint path for the given step.
    """
    game = ConnectFour()
    estimator = ConnectFourNet(learning_rate=1e-4,
                               l2_weight=1e-4, value_weight=0.01,
                               action_indices=game.action_indices)
    checkpoint_name = compute_checkpoint_name(step, checkpoint_path)
    estimator.restore(checkpoint_name)
    return estimator


def train_network(solved_states, evaluate_every):
    print("Converting solved states to training data.")
    training_data = solved_states_to_training_data(solved_states)
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
    trivial_mcts_player = MCTSPlayer(game, trivial_estimator, mcts_iters, 0.5, 0.01)
    rollout_mcts_player = MCTSPlayer(game, rollout_estimator, mcts_iters, 0.5, 0.01)
    # fixed_comparison_players = {1: random_player,
    #                             2: trivial_mcts_player,
    #                             3: rollout_mcts_player}

    fixed_comparison_players = {1: random_player}

    supervised_player_no = len(fixed_comparison_players) + 1
    supervised_players_queue = collections.deque(maxlen=2)

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
        "lr={},batch_size={},value_weight={},l2_weight={},"
        "num_train={}").format(learning_rate, batch_size, value_weight,
                               l2_weight, num_train)

    game_name = 'connect_four-sl'

    current_time_format = time.strftime('%Y-%m-%d_%H:%M:%S')
    path = "experiments/{}-{}-{}/".format(game_name, hyp_string,
                                          current_time_format)
    checkpoint_path = path + 'checkpoints/'
    game_results_file_name = path + "game_results.pickle"

    estimator = ConnectFourNet(learning_rate=learning_rate,
                               l2_weight=l2_weight, value_weight=value_weight,
                               action_indices=game.action_indices)

    summary_path = path + 'logs/'
    scalar_names = ['dev_loss', 'dev_loss_value', 'dev_loss_probs']
    summary_scalars = SummaryScalars(scalar_names)

    verbose = True
    training_iters = -1

    writer = tf.summary.FileWriter(summary_path)

    for step in range(num_steps):
        print("Step: {}".format(step))
        optimise_estimator(estimator, training_data, batch_size,
                           training_iters, mode='supervised', writer=writer,
                           verbose=verbose)

        # Now compute dev loss
        dev_loss, dev_loss_value, dev_loss_probs = estimator.loss(
            dev_data, batch_size)
        print("Dev loss: {}, dev loss value: {}, dev loss probs: {}".format(
            dev_loss, dev_loss_value, dev_loss_probs))

        summary_scalars.run({'dev_loss': dev_loss,
                             'dev_loss_value': dev_loss_value,
                             'dev_loss_probs': dev_loss_probs},
                            estimator.global_step, writer)

        if step % checkpoint_every == 0 and step > 0:
            checkpoint_name = compute_checkpoint_name(step, checkpoint_path)
            estimator.save(checkpoint_name)

            new_estimator = ConnectFourNet(learning_rate=learning_rate,
                                           l2_weight=l2_weight,
                                           value_weight=value_weight,
                                           action_indices=game.action_indices)
            new_estimator.restore(checkpoint_name)

            new_player = MCTSPlayer(game, new_estimator, 500, 0.5)

            supervised_players = {j: player for j, player
                                  in supervised_players_queue}
            comparison_players = {**fixed_comparison_players,
                                  **supervised_players}

            game_results = run_gauntlet(game,
                                        (supervised_player_no, new_player),
                                        comparison_players, 1)

            update_results(game_results, game_results_file_name)

            # elo to writer

            supervised_players_queue.appendleft(
                (supervised_player_no, new_player))
            supervised_player_no += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('training_data', help='Input file with training data.')
    parser.add_argument('--max_lines', help='The maximum number of lines to '
                                            'read in from the training data.')
    parser.add_argument('--evaluate_every',
                        help='The number of epochs between evaluating'
                             'iterations.')
    parser.add_argument('--evaluate_checkpoint_path',
                        help='The checkpoint path to evaluate. If given, '
                             'then evaluate_step must also be provided.')
    parser.add_argument('--evaluate_step',
                        help='The step of the checkpoint to evaluate.')

    args = parser.parse_args()
    training_data = args.training_data

    # Load the training data.
    max_lines = args.max_lines
    if max_lines is not None:
        max_lines = int(max_lines)

    solved_states = []
    with open(training_data, 'r') as f:
        for line in f:
            data = line.split()
            action_list = [int(c) for c in data[0]]
            action = int(data[1])
            outcome = int(data[2])
            solved_states.append((action_list, action, outcome))

            # Break if we have reached max lines.
            if max_lines is not None and len(solved_states) >= max_lines:
                break

    # If evaluate checkpoint path is given, then just evaluate that network.
    if args.evaluate_checkpoint_path is not None:
        checkpoint_path = args.evaluate_checkpoint_path
        checkpoint_step = args.evaluate_step

        estimator = load_net(checkpoint_step, checkpoint_path)

        accuracy = compute_accuracy(estimator, solved_states)
        print("Accuracy: {}".format(accuracy))
    else:
        # Otherwise, train the network.
        evaluate_every = 5
        if args.evaluate_every is not None:
            evaluate_every = int(args.evaluate_every)

        train_network(solved_states, evaluate_every)
