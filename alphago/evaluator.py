from collections import namedtuple, defaultdict
import itertools
from typing import Dict, List, Tuple

from tqdm import tqdm

from .games import Game
from .player import Player

GameLog = namedtuple("GameLog", "result actions game_states".split())
Position, PlayerNo = int, int
PlayerResults = Dict[int, int]


def evaluate(game: Game, players: Dict[Position, Player],
             num_games: int, verbose: bool = True
             ) -> Tuple[PlayerResults, List[GameLog]]:
    """Compare two players. Returns the number of player1 wins,
    losses and draws and the game logs.

    Parameters
    ----------
    game:
        An object representing the game to be played.
    players:
        The players to evaluate against each other, given as dictionary
        with keys position to play in (either 1 or 2) and values the
        players.
    num_games:
        The number of games to be played.
    verbose: optional
        Whether or not to display progress bar during evaluation.

    Returns
    -------
    PlayerResults:
        A dictionary containing the number of wins, losses and draws of
        player1 mapped to keys 1, -1 and 0.
    List[GameLog]:
        A list of logs of the games played, where a game log is a named
        tuple giving the result, actions and game states of a game.
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


def play(game: Game, players: Dict[Position, Player]):
    """Plays a two player game.

    Parameters
    ----------
    game:
        An object representing the game to be played.
    players:
        The players to play the game, given as dictionary with keys
        position to play in (either 1 or 2) and values the players.

    Returns
    -------
    list
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


def run_tournament(game: Game, players: Dict[PlayerNo, Player],
                   num_rounds: int) -> List[Tuple[int, int, float]]:
    """Run a tournament of a the given game between the players and
    return the results.

    Each round constitutes a round-robin, with each player playing
    every other player.

    Parameters
    ----------
    game:
        An object representing the game to be played in the tournament.
    players:
        The players to play in the tournament, given as a dictionary
        mapping player numbers to players.
    num_rounds:
        The number of rounds in the tournament, where in each round,
        every player plays every other player. They play each other
        twice, once in each position i.e. when player i plays player j
        they first play with i as the starting player, and then j.

    Returns
    -------
    list:
        The results of the tournament given in the form of a list of
        (i, j, n) tuples, meaning player i got a score of n against
        player j.
    """
    results = defaultdict(int)
    pairings = tuple(itertools.combinations(players.keys(), 2))
    num_games = num_rounds * 2 * len(pairings)
    with tqdm(total=num_games) as pbar:
        for round_number in range(num_rounds):
            pbar.set_description(f"Round {round_number + 1}")
            for (i, j) in pairings:
                # play i vs j
                pair = {1: players[i], 2: players[j]}
                *_, utility = play(game, pair)
                update_results(i, j, utility, results)
                pbar.update(1)
                # play j vs i
                pair = {2: players[i], 1: players[j]}
                *_, utility = play(game, pair)
                update_results(j, i, utility, results)
                pbar.update(1)

    results_list = [(i, j, n) for (i, j), n in sorted(results.items())]
    return results_list


def run_gauntlet(game: Game, challenger: Tuple[PlayerNo, Player],
                 gauntlet_players: Dict[PlayerNo, Player],
                 num_rounds: int) -> List[Tuple[int, int, float]]:
    """Play a single player against a number of other players, i.e. a
    one vs all tournament, and return the results.

    Parameters
    ----------
    game:
        A object representing the game to played in the gauntlet.
    challenger:
        A tuple containing the player number and player object
        of the challenger.
    gauntlet_players:
        The players to play the challenger against, given by a dict
        mapping player numbers to players.
    num_rounds:
        The number of rounds in the gauntlet, where in each round,
        the challenger plays each of the gauntlet players. They play
        each other twice, once in each position i.e. the challenger
        first plays as the starting player and then as second player.


    Returns
    -------
    list:
        The results of the gauntlet given in the form of a list of
        (i, j, n) tuples, meaning player i got a score of n against
        player j.

    """
    results = defaultdict(int)
    i, player = challenger
    num_games = num_rounds * 2 * len(gauntlet_players)
    with tqdm(total=num_games) as pbar:
        for round_number in range(num_rounds):
            pbar.set_description(f"Round {round_number + 1}")
            for j in gauntlet_players:
                # play i vs j
                pair = {1: player, 2: gauntlet_players[j]}
                *_, utility = play(game, pair)
                update_results(i, j, utility, results)
                pbar.update(1)
                # play j vs i
                pair = {2: player, 1: gauntlet_players[j]}
                *_, utility = play(game, pair)
                update_results(j, i, utility, results)
                pbar.update(1)

    results_list = [(i, j, n) for (i, j), n in sorted(results.items())]
    return results_list


def update_results(player1_no: int, player2_no: int,
                   utility: Dict[int, float],
                   results: Dict[Tuple[int, int], float]) -> None:
    """Update the results given the outcome of a game.

    Parameters
    ----------
    player1_no, player2_no:
        The numbers of two players who played the game. These the player
        numbers used to identify them in the players dict in the
        tournament or gaunlet.
    utility:
        The outcome of game given as a utility dictionary mapping
        players 1 and 2 to either 1, -1, or 0 for a win, loss or draw.
    results:
        The results dictionary to be updated, this maps pairings given
        as tuples of player numbers (i, j) to scores of player i against
        player j.
    """
    # if it is a draw, give each player half a win
    if utility[1] == 0 and utility[2] == 0:
        results[(player1_no, player2_no)] += 0.5
        results[(player2_no, player1_no)] += 0.5
    # otherwise assign the win to the correct player
    elif utility[1] == 1:
        results[(player1_no, player2_no)] += 1
    else:  # utility[2] == 1
        results[(player2_no, player1_no)] += 1
