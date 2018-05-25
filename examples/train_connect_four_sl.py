"""This file trains a connect four net with supervised learning.
"""
import time

import numpy as np
import tensorflow as tf

from alphago.games.connect_four import action_list_to_state, ConnectFour
from alphago.estimator import ConnectFourNet
from alphago.alphago import optimise_estimator



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
        probs_vector[action-1] = 1

        # Store in training_data.
        training_data.append((state, probs_vector, outcome))

    return training_data



if __name__ == "__main__":
    input_file = "tools/connect_four/solver/connect_four_solved_states_chris.txt"

    solved_states = []
    with open(input_file, 'r') as f:
        for line in f:
            data = line.split(' ')
            action_list = [int(c) for c in data[0]]
            action = int(data[1])
            outcome = int(data[2])
            solved_states.append((action_list, action, outcome))

    training_data = solved_states_to_training_data(solved_states)

    learning_rate = 1e-3
    game = ConnectFour()
    game_name = 'connect_four-sl'

    current_time_format = time.strftime('%Y-%m-%d_%H:%M:%S')
    path = "experiments/{}-{}/".format(game_name, current_time_format)
    checkpoint_path = path + 'checkpoints/'
    summary_path = path + 'logs/'

    writer = tf.summary.FileWriter(summary_path)

    estimator = ConnectFourNet(learning_rate=learning_rate,
                               action_indices=game.action_indices)

    num_steps = 1000
    verbose = True
    training_iters = 1000
    batch_size = 32

    for step in range(1000):
        print("Step: {}".format(step))
        optimise_estimator(estimator, training_data, batch_size,
                           training_iters, writer, verbose=verbose)
