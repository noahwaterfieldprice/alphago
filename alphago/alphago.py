import numpy as np
import tqdm

from .utilities import sample_distribution

__all__ = ["train", "play", "self_play_multiple", "build_training_data"]


def train(game, players, action_indices, self_play_iters,
          training_iters, batch_size=32):
    """Runs AlphaGo on the game. 

    Parameters
    ----------

    game: Game
        An object representing the game to be played.
    players: dict of Player
        An dictionary with keys the player numbers and values the players.
    action_indices: dict
        Dictionary with keys the possible actions in the game, and values the
        index of that action.
    self_play_iters: int
        Number of iterations of self-play to run.
    training_iters: int
        Number of training steps to take.
    batch_size: int
    """  # TODO: write better docstring and test this

    all_training_data = []
    losses = []
    with tqdm.tqdm(total=self_play_iters) as pbar:
        for i in range(self_play_iters):
            # Collect training data
            game_states, action_probs = play(game, players)

            training_data = build_training_data(
                game_states, action_probs, game, action_indices)

            # Append to our current training data
            all_training_data.extend(training_data)

            # Only keep the most recent training data
            all_training_data = all_training_data[-10000:]

            # Update tqdm description
            pbar.update(1)

    players[0].estimator.train(all_training_data, batch_size, training_iters)


#
# def train_alphago_estimator(num_train_steps, training_data, batch_size):
#     # Don't train if we don't have enough training data for a batch.
#     # TODO: Move batch_size to training function.
#     if len(all_training_data) < batch_size:
#         return
#
#     with tqdm.tqdm(total=num_train_steps) as pbar:
#
#         for i in range(num_train_steps):
#             # Train on the data
#             batch_indices = np.random.choice(len(all_training_data),
#                                              batch_size,
#                                              replace=True)
#             train_batch = [all_training_data[ix] for ix in
#                            batch_indices]
#             loss = train_function(train_batch)
#             losses.append(loss)
#             pbar.set_description("Avg loss: {0:.5f}".format(np.mean(losses)))
#
#             # Update tqdm description
#             if i % 100 == 0:
#                 pbar.update(100)


def self_play(game, player):
    pass




def play(game, players):
    """Plays a two player game.

    Parameters
    ----------
    game: Game
        An object representing the game to be played.
    players: dict of Player
        An dictionary with keys the player numbers and values the players.
        
    Returns
    -------
    game_state_list: list
        A list of game states encountered in the self-play game. Starts
        with the initial state and ends with a terminal state.
    action_probs_list: list
        A list of action probability dictionaries, as returned by MCTS
        each time the algorithm has to take an action. The ith action
        probabilities dictionary corresponds to the ith game_state, and
        action_probs_list has length one less than game_state_list,
        since we don't have to move in a terminal state.
    """
    game_state = game.INITIAL_STATE
    game_state_list = [game_state]
    action_probs_list = []

    while not game.is_terminal(game_state):
        # First run MCTS to compute action probabilities.
        player_no = game.which_player(game_state)
        action_probs = players[player_no].action_probabilities(game_state)

        # Choose the action according to the action probabilities.
        action = sample_distribution(action_probs)

        # Play the action
        next_states = game.compute_next_states(game_state)
        game_state = next_states[action]

        # Add the action probabilities and game state to the list.
        action_probs_list.append(action_probs)
        game_state_list.append(game_state)

    return game_state_list, action_probs_list


def build_training_data(states_, action_probs_, game, action_indices):
    """Takes a list of states and action probabilities, as returned by
    play, and creates training data from this. We build up a list
    consisting of (state, probs, z) tuples, where player is the player
    in state 'state', and 'z' is the utility to 'player' in 'last_state'.

    We omit the terminal state from the list as there are no probabilities to
    train. TODO: Potentially include the terminal state in order to train the
    value.

    Parameters
    ----------
    states_: list
        A list of n states, with the last being terminal.
    action_probs_: list
        A list of n-1 dictionaries containing action probabilities. The ith
        dictionary applies to the ith state, representing the probabilities
        returned by play of taking each available action in the state.
    game: Game
        An object representing the game to be played.
    action_indices: dict
        A dictionary mapping actions (in the form of the compute_next_states
        function) to action indices (to be used for training the neural
        network).

    Returns
    -------
    training_data: list
        A list consisting of (state, probs, z) tuples, where player is the
        player in state 'state', and 'z' is the utility to 'player' in
        'last_state'.
    """

    # Get the outcome for the game. This should be the last state in states_.
    last_state = states_.pop()
    outcome = game.utility(last_state)

    # Now action_probs_ and states_ are the same length.
    training_data = []
    for state, probs in zip(states_, action_probs_):
        # Get the player in the state, and the value to this player of the
        # terminal state.
        player = game.which_player(state)
        z = outcome[player]

        # Convert the probs dictionary to a numpy array using action_indices.
        probs_vector = np.zeros(len(action_indices))
        for action, prob in probs.items():
            probs_vector[action_indices[action]] = prob

        non_nan_state = np.nan_to_num(state)

        training_data.append((non_nan_state, probs_vector, z))

    return training_data


def self_play_multiple(game, evaluator, action_indices, mcts_iters,
                       c_puct, num_self_play):
    """Combines play and build_training_data to generate training data
    given a game and an evaluator.

    Parameters
    ----------
    game: Game
        An object representing the game to be played.
    evaluator: func
        An evaluator.
    action_indices: dict
        Dictionary with keys the actions and values an index for the action.
    mcts_iters: int
        Number of iterations to run MCTS for.
    c_puct: float
        Parameter for MCTS.
    num_self_play: int
        Number of games to play in 'self-play'

    Returns
    -------
    training_data: list
        A list of training data tuples. See 'build_training_data'.
    """

    training_data = []
    for i in range(num_self_play):
        game_states_, action_probs_ = play(
            game, evaluator, mcts_iters, c_puct)
        training_data.append(build_training_data(
            game_states_, action_probs_, game, action_indices))
    return training_data
