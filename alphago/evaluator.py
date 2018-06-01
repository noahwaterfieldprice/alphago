from collections import namedtuple, defaultdict
import itertools
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from .games import Game
from .player import Player

GameLog = namedtuple("GameLog", "result actions game_states".split())


def evaluate(game: Game, players, num_games, verbose=True):
    """Compare two evaluators. Returns the number of evaluator1 wins
    and number of draws in the games, as well as the total number of
    games.
    """

    win, loss, draw = 1, -1, 0
    player1_results = {win: 0, loss: 0, draw: 0}
    game_logs = []

    if verbose:
        disable_tqdm = False
    else:
        disable_tqdm = True

    with tqdm(total=num_games, disable=disable_tqdm) as pbar:
        for game_no in range(num_games):
            for player in players.values():
                player.reset()

            actions, game_states, utility = play(game, players)

            player1_result = utility[1]
            player1_results[player1_result] += 1

            pbar.update(1)
            pbar.set_description("Win1/Win2/Draw: {}/{}/{}".format(
                player1_results[win], player1_results[loss],
                player1_results[draw]))

            game_logs.append(GameLog(player1_result, actions, game_states))

    return player1_results, game_logs


def play(game: Game, players: Dict[int, Player]):
    """Plays a two player game.

    Parameters
    ----------
    game:
        An object representing the game to be played.
    players:
        An dictionary with keys the player numbers and values the
        players.

    Returns
    -------
    game_state_list: list
        A list of game states encountered in the self-play game. Starts
        with the initial state and ends with a terminal state.
    """
    game_state = game.initial_state
    game_states = [game_state]
    for player in players.values():
        player.reset()
    actions = []

    while not game.is_terminal(game_state):
        # first run MCTS to compute action probabilities.
        player_no = game.which_player(game_state)
        action = players[player_no].choose_action(game_state)

        # play the action
        next_states = game.compute_next_states(game_state)
        game_state = next_states[action]

        # update players with the played action
        for player in players.values():
            player.update(action)

        actions.append(action)
        game_states.append(game_state)

    utility = game.utility(game_states[-1])
    return actions, game_states, utility


def run_tournament(game: Game, players: Dict[int, Player],
                   num_rounds: int) -> List[Tuple]:
    """Run a tournament of a the given game between the players and
    return the results.

    Each round constitutes a round-robin, with each player playing
    every other player.

    Parameters
    ----------
    game:
        An object representing the game to be played.
    players:
        A dictionary mapping player
    num_rounds:

    """
    results = defaultdict(int)
    pairings = tuple(itertools.combinations(players.keys(), 2))
    for round_number in range(num_rounds):
        with tqdm(total=len(pairings)) as pbar:
            pbar.set_description(f"Round {round_number}")
            for (i, j) in pairings:
                # play i vs j
                pair = {1: players[i], 2: players[j]}
                *_, utility = play(game, pair)
                update_results(i, j, utility, results)
                # play j vs i
                pair = {2: players[i], 1: players[j]}
                *_, utility = play(game, pair)
                update_results(j, i, utility, results)
                pbar.update(1)

    results_list = [(i, j, n) for (i, j), n in sorted(results.items())]
    return results_list


def update_results(player1, player2, utility, results):
    """Update the wins matrix given the outcome of a game."""
    # if it is a draw, give each player half a win
    if utility[1] == 0 and utility[2] == 0:
        results[(player1, player2)] += 0.5
        results[(player2, player1)] += 0.5
    # otherwise assign the win to the correct player
    elif utility[1] == 1:
        results[(player1, player2)] += 1
    else:
        results[(player2, player1)] += 1
