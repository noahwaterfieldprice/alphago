"""This file trains a connect four net with supervised learning.
"""
import argparse
from collections import deque
import os
import pickle
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
        (state, actions, outcome), where state is a connect four state;
        actions are the optimal actions to play in the state, and outcome is
        the eventual outcome to the current player.

    Returns
    -------
    training_data: list
        A list of training data suitable for AlphaGo.
    """
    training_data = []
    for state, actions, outcome in solved_states:
        # Set the probs vector to be 1 for the optimal actions, and 0 for all
        # other actions, but normalise so it sums to 1.
        probs_vector = np.array([1 / len(actions) if a + 1 in actions else 0
                                 for a in range(7)])

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

    return np.mean([1 if predicted_actions[i] in actions_list[i]
                    else 0 for i in range(len(predicted_actions))])


# Training data should be a file with lines of the form:
# action_list value optimal_actions
# where optimal actions is a space separated sequence of the optimal actions
# in that position. All actions should be indexed 1 to 7.
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
    c_puct = 0.5
    trivial_mcts_player = MCTSPlayer(game, trivial_estimator, mcts_iters,
                                     c_puct, 0.01)
    rollout_mcts_player = MCTSPlayer(game, rollout_estimator, mcts_iters,
                                     c_puct, 0.01)
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
    scalar_names = ['dev_loss', 'dev_loss_value', 'dev_loss_probs',
                    'dev_accuracy']
    summary_scalars = SummaryScalars(scalar_names)

    verbose = True
    training_iters = -1

    writer = tf.summary.FileWriter(summary_path)

    dev_optimal_actions = [(state, optimal_actions) for
                           state, optimal_actions, value in dev_data]

    for step in range(num_steps):
        print("Step: {}".format(step))
        optimise_estimator(estimator, training_data, batch_size,
                           training_iters, mode='supervised', writer=writer,
                           verbose=verbose)

        # Now compute dev loss
        dev_loss, dev_loss_value, dev_loss_probs = estimator.loss(
            dev_data, batch_size)
        dev_accuracy = compute_accuracy(estimator, dev_optimal_actions)
        print("Dev loss: {}, dev loss value: {}, dev loss probs: {}, "
              "dev accuracy: {}".format(dev_loss, dev_loss_value,
                                        dev_loss_probs, dev_accuracy))

        summary_scalars.run({'dev_loss': dev_loss,
                             'dev_loss_value': dev_loss_value,
                             'dev_loss_probs': dev_loss_probs,
                             'dev_accuracy': dev_accuracy},
                            estimator.global_step, writer)

        if step % checkpoint_every == 0 and step > 0:
            checkpoint_name = compute_checkpoint_name(step, checkpoint_path)
            estimator.save(checkpoint_name)

            new_estimator = ConnectFourNet(learning_rate=learning_rate,
                                           l2_weight=l2_weight,
                                           value_weight=value_weight,
                                           action_indices=game.action_indices)
            new_estimator.restore(checkpoint_name)

            new_player = MCTSPlayer(game, new_estimator, mcts_iters, c_puct)

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


def split_solved_state(line):
    """Splits a line from a solved states file into the action list,
    value and optimal actions.

    Parameters
    ----------
    s: str
        A line from a solved states file.

    Returns
    -------
    state: ndarray
        Numpy array representing the state.
    optimal_actions: list
        List of the optimal actions.
    value: int
        The value of the state. Equals 1 if player can force a win,
        0 if player can force a draw and -1 if opponent can force a win.
    """
    data = line.strip().split(',')
    action_list = list(map(int, data[0]))
    state = action_list_to_state([a - 1 for a in action_list])
    value = int(data[1])
    optimal_actions = list(map(int, data[2].split(' ')))
    return state, optimal_actions, value


def load_solved_states(training_data_file, max_lines=None):
    solved_states = []
    with open(training_data_file, 'r') as f:
        for line in f:
            state, optimal_actions, value = split_solved_state(line)
            solved_states.append((state, optimal_actions, value))

            # Break if we have reached max lines.
            if max_lines is not None and len(solved_states) >= max_lines:
                break

    return solved_states


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

    # Load the training data.
    max_lines = int(args.max_lines) if args.max_lines is not None else None

    solved_states = load_solved_states(args.training_data, max_lines=max_lines)

    # If evaluate checkpoint path is given, then just evaluate that network.
    if args.evaluate_checkpoint_path is not None:
        checkpoint_path = args.evaluate_checkpoint_path
        checkpoint_step = args.evaluate_step

        estimator = load_net(checkpoint_step, checkpoint_path)

        optimal_actions_list = [(state, optimal_actions)
                                for state, optimal_actions, _ in solved_states]

        accuracy = compute_accuracy(estimator, optimal_actions_list)
        print("Accuracy: {}".format(accuracy))
    else:
        # Otherwise, train the network.
        evaluate_every = 5
        if args.evaluate_every is not None:
            evaluate_every = int(args.evaluate_every)

        train_network(solved_states, evaluate_every)
