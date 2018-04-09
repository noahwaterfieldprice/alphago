import numpy as np
from tqdm import tqdm

from alphago.evaluator import play
from .player import MCTSPlayer
from .evaluator import evaluate
from .mcts_tree import MCTSNode, mcts

__all__ = ["train_alphago", "train", "self_play", "build_training_data"]


def train_alphago(game, create_estimator, self_play_iters, training_iters,
                  checkpoint_path, alphago_steps=100, evaluate_every=1,
                  batch_size=32, mcts_iters=100, c_puct=1.0,
                  replay_length=100000, num_evaluate_games=500,
                  win_rate=0.55):
    """Trains AlphaGo on the game.

    Parameters
    ----------
    game: object
        An object that has the attributes a game needs.
    create_estimator: func
        Creates a trainable estimator for the game. The estimator should
        have a train function.
    self_play_iters: int
        Number of self-play games to play each self-play step.
    training_iters: int
        Number of training iters to use for each training step.
    checkpoint_path: str
        Where to save the checkpoints to.
    alphago_steps: int
        Number of steps to run the alphago loop for.
    evaluate_every: int
        Evaluate the network every evaluate_every steps.
    batch_size: int
        Batch size to train with.
    mcts_iters: int
        Number of iterations to run MCTS for.
    c_puct: float
        Parameter for MCTS. See AlphaGo paper.
    replay_length: int
        The amount of training data to use. Only train on the most recent
        training data.
    evaluator_games: int
        Number of games to evaluate the players for.
    win_rate: float
        Number between 0 and 1. Only update self-play player when training
        player beats self-play player by at least this rate.
    """
    # TODO: Do self-play, training and evaluating in parallel.

    # We use a fixed estimator (the best one that's been trained) to
    # generate self-play training data. We then train the training estimator
    # on that data. We produce a checkpoint every 1000 training steps. This
    # checkpoint is then evaluated against the current best neural network.
    # If it beats the current best network by at least 55% then it becomes
    # the new best network.
    # 1 is the fixed player, and 2 is the training player.
    self_play_player = create_estimator()
    training_player = create_estimator()

    checkpoint_names = []

    all_losses = []

    all_training_data = []
    for alphago_step in range(alphago_steps):
        print("Self-play")
        # Collect self-play training data using the best estimator.
        for _ in tqdm(range(self_play_iters)):
            game_state_list, action_probs_list = self_play(
                game, self_play_player.create_estimate_fn(), mcts_iters, c_puct)
            training_data = build_training_data(game_state_list,
                                                action_probs_list, game,
                                                game.ACTION_INDICES)

        # Append training data
        all_training_data.extend(training_data)
        # Only keep most recent data
        all_training_data = all_training_data[-replay_length:]

        # Train the training player on self-play training data.
        print("Training")
        losses = training_player.train(all_training_data, batch_size,
                                       training_iters)
        all_losses.append(losses)
        print("Mean loss: {}".format(np.mean(losses)))

        # Evaluate the players and choose the best.
        if alphago_step % evaluate_every == 0:
            # Checkpoint the model.
            # TODO: Implement evaluation
            # TODO: Refactor so the MCTSPlayer doesn't need to know player
            # number. It should be able to play in both positions.

            print("Evaluating")
            wins1, wins2, draws = evaluate_in_both_positions(
                game, self_play_player.create_estimate_fn(),
                training_player.create_estimate_fn(), mcts_iters, c_puct,
                num_evaluate_games)

            print("Self-play player wins: {}, Training player wins: {}, "
                  "Draws: {}".format(wins1, wins2, draws))
            training_win_rate = wins2 / (wins1 + wins2 + draws)
            print("Win rate for training player: {}".format(
                training_win_rate))

            # Checkpoint the training player.
            checkpoint_name = checkpoint_path + "{}.checkpoint".format(
                alphago_step)
            checkpoint_names.append(checkpoint_name)
            print("Saving at: {}".format(checkpoint_name))
            training_player.save(checkpoint_name)

            # If training player beats self-play player by a large enough
            # margin, then it becomes the new best estimator.
            if training_win_rate > win_rate:
                # Create a new self player, with the weights of the most
                # recent training_player.
                print("Updating self-play player.")
                self_play_player = create_estimator()
                print("Restoring from: {}".format(checkpoint_name))
                self_play_player.restore(checkpoint_name)

    return


def evaluate_in_both_positions(game, estimator1, estimator2, mcts_iters,
                               c_puct, num_evaluate_games):
    # Evaluate estimator1 vs estimator2.
    players = {1: MCTSPlayer(1, game, estimator1, mcts_iters, c_puct),
               2: MCTSPlayer(2, game, estimator2, mcts_iters, c_puct)}
    player1_results = evaluate(game, players, num_evaluate_games)
    wins1 = player1_results[1]
    wins2 = player1_results[-1]
    draws = player1_results[0]

    # Evaluate estimator2 vs estimator1.
    players = {1: MCTSPlayer(1, game, estimator2, mcts_iters, c_puct),
               2: MCTSPlayer(2, game, estimator1, mcts_iters, c_puct)}
    player1_results = evaluate(game, players, num_evaluate_games)
    wins1 += player1_results[-1]
    wins2 += player1_results[1]
    draws += player1_results[0]

    return wins1, wins2, draws


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
    with tqdm(total=self_play_iters) as pbar:
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
#     with tqdm(total=num_train_steps) as pbar:
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


def self_play(game, estimator, mcts_iters, c_puct):
    """Plays a game using MCTS to choose actions for both players.
    Parameters
    ----------
    game: Game
        An object representing the game to be played.
    estimator: func
        An estimate function.
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
        action_probs = mcts(node, game, estimator, mcts_iters, c_puct)

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
