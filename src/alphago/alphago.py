from collections import OrderedDict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .evaluator import evaluate
from .mcts_tree import MCTSNode, mcts
from .metric_logger import NullLogger
from .player import MCTSPlayer, OptimalPlayer, RandomPlayer
from .utilities import sample_distribution

__all__ = [
    "train_alphago",
    "self_play",
    "process_self_play_data",
    "process_training_data",
]


def compute_checkpoint_name(step, path):
    return str(Path(path) / f"{step}.pt")


def train_alphago(game, create_estimator, cfg, logger=None):
    """Trains AlphaGo on the game.

    Parameters
    ----------
    game: object
        An object that has the attributes a game needs.
    create_estimator: func
        Creates a trainable estimator for the game. The estimator should
        have a train function.
    cfg: Config
        The resolved experiment configuration. Every scalar hyperparameter is
        read from this object (``cfg.training.*``, ``cfg.mcts.*``,
        ``cfg.paths.*``, ``cfg.verbose``) and only plain primitives are passed
        below the estimator/MCTS seam.
    logger: MetricLogger or None
        A metric sink exposing ``log_scalar(tag, value, step)`` and ``close()``.
        ``None`` resolves to a :class:`NullLogger`, so the loop can log losses
        and eval rates unconditionally.
    """
    # TODO: Do self-play, training and evaluating in parallel.
    logger = logger or NullLogger()

    # Extract every scalar from cfg into plain locals. Only these primitives
    # (ints/floats/strings) cross the estimator/MCTS seam below — never cfg or
    # a cfg subgroup.
    self_play_iters = cfg.training.self_play_iters
    training_iters = cfg.training.training_iters
    alphago_steps = cfg.training.alphago_steps
    evaluate_every = cfg.training.evaluate_every
    batch_size = cfg.training.batch_size
    replay_length = cfg.training.replay_length
    num_evaluate_games = cfg.training.num_evaluate_games
    win_rate = cfg.training.win_rate
    restore_dir = cfg.training.restore_dir
    restore_step = cfg.training.restore_step
    mcts_iters = cfg.mcts.mcts_iters
    c_puct = cfg.mcts.c_puct
    checkpoint_path = cfg.paths.checkpoint_dir
    verbose = cfg.verbose

    # Ensure the checkpoint directory exists before the first write. This covers
    # both the train_alphago and comparator entry points, so the first
    # torch.save cannot raise FileNotFoundError.
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    # We use a fixed estimator (the best one that's been trained) to
    # generate self-play training data. We then train the training estimator
    # on that data. We produce a checkpoint every 1000 training steps. This
    # checkpoint is then evaluated against the current best neural network.
    # If it beats the current best network by at least 55% then it becomes
    # the new best network.
    # 1 is the fixed player, and 2 is the training player.
    self_play_estimator = create_estimator()
    training_estimator = create_estimator()

    # Cross-run restore: read from the PREVIOUS run's checkpoint dir
    # (restore_dir) when set, while checkpoints below write the fresh run's
    # checkpoint_path.
    # Fail loud: a restore_dir without a restore_step would silently
    # no-op the restore (the block below only fires when restore_step is set),
    # so surface the misconfiguration instead of quietly starting from scratch.
    if restore_dir is not None and restore_step is None:
        raise RuntimeError(
            "cfg.training.restore_dir is set but cfg.training.restore_step is "
            "None, so no checkpoint would be restored (the restore silently "
            "no-ops). Set cfg.training.restore_step to the step to restore "
            "from, or clear restore_dir to start training from scratch."
        )
    if restore_step is not None:
        restore_source = restore_dir if restore_dir is not None else checkpoint_path
        restore_path = compute_checkpoint_name(restore_step, restore_source)
        self_play_estimator.restore(restore_path)
        training_estimator.restore(restore_path)

    all_losses = []
    self_play_data = None

    initial_step = restore_step + 1 if restore_step is not None else 0
    # Run the step loop under try/finally so logger.close() releases the
    # metric sink (e.g. flushes/closes the wandb run) even if a step raises,
    # rather than leaking it on the failure path.
    try:
        for alphago_step in range(initial_step, initial_step + alphago_steps):
            self_play_data = generate_self_play_data(
                game,
                self_play_estimator,
                mcts_iters,
                c_puct,
                self_play_iters,
                verbose=verbose,
                data=self_play_data,
            )

            training_data = process_training_data(
                self_play_data, replay_length, verbose=verbose
            )
            if len(training_data) < 100:
                continue
            summary = optimise_estimator(
                training_estimator,
                training_data,
                batch_size,
                training_iters,
                verbose=verbose,
            )

            # Log the loss components on the alphago_step axis. Guard
            # against a None summary (no training step ran on this pass).
            if summary is not None:
                logger.log_scalar("loss/total", summary["total"], step=alphago_step)
                logger.log_scalar("loss/policy", summary["policy"], step=alphago_step)
                logger.log_scalar("loss/value", summary["value"], step=alphago_step)
                # Record each step's total loss so the return value is a
                # real loss history rather than an always-empty list.
                all_losses.append(summary["total"])

            # Evaluate the players and choose the best.
            if alphago_step % evaluate_every == 0:
                success_rate, success_rate_random = evaluate_model(
                    game,
                    self_play_estimator,
                    training_estimator,
                    mcts_iters,
                    c_puct,
                    num_evaluate_games,
                    verbose=verbose,
                )

                # Log eval success rates on the same alphago_step axis.
                logger.log_scalar("eval/success_rate", success_rate, step=alphago_step)
                logger.log_scalar(
                    "eval/success_rate_random", success_rate_random, step=alphago_step
                )

                checkpoint_model(training_estimator, alphago_step, checkpoint_path)

                # If training player beats self-play player by a large enough
                # margin, then it becomes the new best estimator.
                if success_rate > win_rate:
                    # Create a new self player, with the weights of the most
                    # recent training_estimator.
                    if verbose:
                        print("Updating self-play player.")
                        print(f"Restoring from step: {alphago_step}")
                    self_play_estimator = create_estimator()
                    restore_path = compute_checkpoint_name(
                        alphago_step, checkpoint_path
                    )
                    self_play_estimator.restore(restore_path)
    finally:
        logger.close()

    return all_losses


def optimise_estimator(
    estimator,
    training_data,
    batch_size,
    training_iters,
    mode="reinforcement",
    verbose=True,
):
    summary = estimator.train(
        training_data,
        batch_size,
        training_iters,
        mode=mode,
        verbose=verbose,
    )
    return summary


def evaluate_model(game, player1, player2, mcts_iters, c_puct, num_games, verbose=True):
    # Checkpoint the model.
    # TODO: Implement evaluation
    # TODO: Choose tau more systematically.

    if verbose:
        print("Evaluating. Self-player vs training, then training vs self-player")
    wins1, wins2, draws = evaluate_estimators_in_both_positions(
        game,
        player1.create_estimate_fn(),
        player2.create_estimate_fn(),
        mcts_iters,
        c_puct,
        num_games,
        tau=0.01,
        verbose=verbose,
    )

    if verbose:
        print(
            f"Self-play player wins: {wins1}, "
            f"Training player wins: {wins2}, Draws: {draws}"
        )

    success_rate = (wins2 + draws) / (wins1 + wins2 + draws)
    if verbose:
        print(f"Win + draw rate for training player: {success_rate}")

    # Also evaluate against a random player
    wins1, wins2, draws = evaluate_mcts_against_random_player(
        game,
        player2.create_estimate_fn(),
        mcts_iters,
        c_puct,
        num_games,
        tau=0.01,
        verbose=verbose,
    )
    success_rate_random = (wins1 + draws) / (wins1 + wins2 + draws)

    if verbose:
        print(
            f"Training player vs random. Wins: {wins1}, Losses: {wins2}, Draws: {draws}"
        )

    return success_rate, success_rate_random


def checkpoint_model(player, step, path):
    """Checkpoint the training player."""
    checkpoint_name = compute_checkpoint_name(step, path)
    player.save(checkpoint_name)


def evaluate_mcts_against_optimal_player(
    game, estimator, mcts_iters, c_puct, num_evaluate_games, tau, verbose=True
):
    # Evaluate estimator1 vs estimator2.
    players = {
        1: MCTSPlayer(game, estimator, mcts_iters, c_puct, tau=tau),
        2: OptimalPlayer(game),
    }
    player1_results, _ = evaluate(game, players, num_evaluate_games, verbose=verbose)
    wins1 = player1_results[1]
    wins2 = player1_results[-1]
    draws = player1_results[0]

    # Evaluate estimator2 vs estimator1.
    players = {
        1: OptimalPlayer(game),
        2: MCTSPlayer(game, estimator, mcts_iters, c_puct, tau=tau),
    }
    player1_results, _ = evaluate(game, players, num_evaluate_games, verbose=verbose)
    wins1 += player1_results[-1]
    wins2 += player1_results[1]
    draws += player1_results[0]

    return wins1, wins2, draws


def evaluate_mcts_against_random_player(
    game, estimator, mcts_iters, c_puct, num_evaluate_games, tau, verbose=True
):
    # Evaluate estimator1 vs estimator2.
    players = {
        1: MCTSPlayer(game, estimator, mcts_iters, c_puct, tau=tau),
        2: RandomPlayer(game),
    }
    player1_results, _ = evaluate(game, players, num_evaluate_games, verbose=verbose)
    wins1 = player1_results[1]
    wins2 = player1_results[-1]
    draws = player1_results[0]

    # Evaluate estimator2 vs estimator1.
    players = {
        1: RandomPlayer(game),
        2: MCTSPlayer(game, estimator, mcts_iters, c_puct, tau=tau),
    }
    player1_results, _ = evaluate(game, players, num_evaluate_games, verbose=verbose)
    wins1 += player1_results[-1]
    wins2 += player1_results[1]
    draws += player1_results[0]

    return wins1, wins2, draws


def evaluate_estimators_in_both_positions(
    game,
    estimator1,
    estimator2,
    mcts_iters,
    c_puct,
    num_evaluate_games,
    tau,
    verbose=True,
):
    # Evaluate estimator1 vs estimator2.
    players = {
        1: MCTSPlayer(game, estimator1, mcts_iters, c_puct, tau=tau),
        2: MCTSPlayer(game, estimator2, mcts_iters, c_puct, tau=tau),
    }
    player1_results, _ = evaluate(game, players, num_evaluate_games, verbose=verbose)
    wins1 = player1_results[1]
    wins2 = player1_results[-1]
    draws = player1_results[0]

    # Evaluate estimator2 vs estimator1.
    players = {
        1: MCTSPlayer(game, estimator2, mcts_iters, c_puct, tau=tau),
        2: MCTSPlayer(game, estimator1, mcts_iters, c_puct, tau=tau),
    }
    player1_results, _ = evaluate(game, players, num_evaluate_games, verbose=verbose)
    wins1 += player1_results[-1]
    wins2 += player1_results[1]
    draws += player1_results[0]

    return wins1, wins2, draws


def generate_self_play_data(
    game, estimator, mcts_iters, c_puct, num_iters, data=None, verbose=True
):
    """Generates self play data for a number of iterations for a given
    estimator. Saves to save_file_path, if given.
    """
    if data is not None:
        index = max(data.keys()) + 1
    else:
        data = OrderedDict()
        index = 0

    # Collect self-play training data using the best estimator.
    disable_tqdm = not verbose
    for _ in tqdm(range(num_iters), disable=disable_tqdm):
        data[index] = self_play(
            game, estimator.create_estimate_fn(), mcts_iters, c_puct
        )
        index += 1

    return data


def self_play(game, estimator, mcts_iters, c_puct):
    """Plays a single game using MCTS to choose actions for both players.

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
    probs_list: list
        A list of action probability dictionaries, as returned by MCTS
        each time the algorithm has to take an action. The ith action
        probabilities dictionary corresponds to the ith game_state, and
        probs_list has length one less than game_state_list,
        since we don't have to move in a terminal state.
    """
    node = MCTSNode(game.initial_state, game.current_player(game.initial_state))

    game_state_list = [node.game_state]
    probs_list = []
    action_list = []

    move_count = 0

    while not node.is_terminal:
        # TODO: Choose this better.
        tau = 1
        if move_count >= 10:
            tau = 1 / (move_count - 10 + 1)

        # First run MCTS to compute action probabilities.
        action_probs = mcts(node, game, estimator, mcts_iters, c_puct, tau=tau)

        # Choose the action according to the action probabilities.
        action = sample_distribution(action_probs)
        action_list.append(action)

        # Play the action
        node = node.children[action]

        # Add the action probabilities and game state to the list.
        probs_list.append(action_probs)
        game_state_list.append(node.game_state)
        move_count += 1

    data = process_self_play_data(
        game_state_list, action_list, probs_list, game, game.action_indices
    )

    return data


def process_training_data(self_play_data, replay_length=None, verbose=True):
    """Takes self play data and returns a list of tuples (state,
    action_probs, utility) suitable for training an estimator.

    Parameters
    ----------
    self_play_data: dict
        Dictionary with keys given by an index (int) and values given by a
        log of the game. This is a list of tuples as in generate self play
        data.
    replay_length: int or None
        If given, only return the last replay_length (state, probs, utility)
        tuples.
    verbose: bool
        If True, print the training-data and self-play-data lengths. Gated so
        the training loop stays quiet unless verbose output is requested
       , mirroring the evaluator's verbose convention.
    """
    training_data = []
    for game_log in self_play_data.values():
        for state, _action, probs_vector, z in game_log:
            training_data.append((state, probs_vector, z))

    if verbose:
        print(f"Training data length: {len(training_data)}")
        print(f"Self play data length: {len(self_play_data)}")

    if replay_length is not None:
        training_data = training_data[-replay_length:]

    return training_data


def process_self_play_data(states, actions, action_probs, game, action_indices):
    """Takes a list of states and action probabilities, as returned by
    play, and creates training data from this. We build up a list
    consisting of (state, probs, z) tuples, where player is the player
    in state 'state', and 'z' is the utility to 'player' in 'last_state'.

    We omit the terminal state from the list as there are no probabilities to
    train. TODO: Potentially include the terminal state in order to train the
    value.

    Parameters
    ----------
    states: list
        A list of n states, with the last being terminal.
    actions: list
        A list of n-1 actions, being the action taken in the corresponding
        state.
    action_probs: list
        A list of n-1 dictionaries containing action probabilities. The ith
        dictionary applies to the ith state, representing the probabilities
        returned by play of taking each available action in the state.
    game: Game
        An object representing the game to be played.
    action_indices: dict
        A dictionary mapping actions (in the form of the legal_actions
        function) to action indices (to be used for training the neural
        network).

    Returns
    -------
    training_data: list
        A list consisting of (state, action, probs, z) tuples, where player
        is the player in state 'state', and 'z' is the utility to 'player' in
        'last_state'.
    """

    # Get the outcome for the game. This should be the last state in states.
    last_state = states[-1]
    outcome = game.utility(last_state)

    # Now action_probs and states are the same length.
    training_data = []
    for state, action, probs in zip(states, actions, action_probs, strict=False):
        # Get the player in the state, and the value to this player of the
        # terminal state.
        player = game.current_player(state)
        z = outcome[player]

        # Convert the probs dictionary to a numpy array using action_indices.
        probs_vector = np.zeros(len(action_indices))
        for a, prob in probs.items():
            probs_vector[action_indices[a]] = prob

        non_nan_state = np.nan_to_num(state)

        training_data.append((non_nan_state, action, probs_vector, z))

    return training_data
