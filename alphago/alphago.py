import numpy as np
import tqdm

from . import mcts, MCTSNode

__all__ = ["train", "self_play", "self_play_multiple", "build_training_data"]


def train(evaluator, train_function, action_indices, game,
            self_play_iters, num_train_steps, mcts_iters, c_puct,
            batch_size=32):
    # TODO: write better docstring
    """Runs AlphaGo on the game.

    Parameters
    ----------
    evaluator: func
        An evaluator.
    train_function: func
        A function to train the evaluator. Takes in 'training_data' as input and
        outputs the training loss.
    action_indices: dict
        Dictionary with keys the possible actions in the game, and values the
        index of that action.
    game: Game
        An object representing the game to be played.
    self_play_iters: int
        Number of iterations of self-play to run.
    num_train_steps: int
        Number of training steps to take.
    mcts_iters: int
        Number of iterations to run MCTS for.
    c_puct: float
        Parameter for MCTS.
    """

    all_training_data = []
    losses = []
    with tqdm.tqdm(total=self_play_iters) as pbar:
        for i in range(self_play_iters):
            # Collect training data
            game_states, action_probs = self_play(
                game, evaluator, mcts_iters, c_puct)

            training_data = build_training_data(
                game_states, action_probs, game, action_indices)

            # Append to our current training data
            all_training_data.extend(training_data)

            # Only keep the most recent training data
            all_training_data = all_training_data[-10000:]

            # Update tqdm description
            pbar.update(1)

    # Don't train if we don't have enough training data for a batch.
    # TODO: Move batch_size to training function.
    if len(all_training_data) < batch_size:
        return

    with tqdm.tqdm(total=num_train_steps) as pbar:
        
        for i in range(num_train_steps):
            # Train on the data
            batch_indices = np.random.choice(len(all_training_data),
                                             batch_size,
                                             replace=True)
            train_batch = [all_training_data[ix] for ix in
                           batch_indices]
            loss = train_function(train_batch)
            losses.append(loss)
            pbar.set_description("Avg loss: {0:.5f}".format(np.mean(losses)))

            # Update tqdm description
            if i % 100 == 0:
                pbar.update(100)


def self_play(game, evaluator, mcts_iters, c_puct):
    """Plays a game using MCTS to choose actions for both players.

    Parameters
    ----------
    game: Game
        An object representing the game to be played.
    evaluator: func
        An evaluator.
    mcts_iters: int
        Number of iterations to run MCTS for.
    c_puct: float
        Parameter for MCTS.

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
    node = MCTSNode(game.INITIAL_STATE, game.which_player(game.INITIAL_STATE))

    game_state_list = [node.game_state]
    action_probs_list = []

    while not node.is_terminal:
        # First run MCTS to compute action probabilities.
        action_probs = mcts(node, game, evaluator, mcts_iters, c_puct)

        # Choose the action according to the action probabilities.
        actions, probs = zip(*action_probs.items())
        action_ix = np.random.choice(len(actions), p=probs)
        action = actions[action_ix]

        # Play the action
        node = node.children[action]

        # Add the action probabilities and game state to the list.
        action_probs_list.append(action_probs)
        game_state_list.append(node.game_state)

    return game_state_list, action_probs_list


def build_training_data(states_, action_probs_, game, action_indices):
    """Takes a list of states and action probabilities, as returned by
    self_play, and creates training data from this. We build up a list
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
        returned by self_play of taking each available action in the state.
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
    """Combines self_play and build_training_data to generate training data
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
        game_states_, action_probs_ = self_play(
            game, evaluator, mcts_iters, c_puct)
        training_data.append(build_training_data(
            game_states_, action_probs_, game, action_indices))
    return training_data
