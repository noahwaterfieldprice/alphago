from typing import Dict, Sequence, Tuple

import numpy as np


def compute_player_indices(game_results):
    """Computes a dictionary for players with keys the player and values an
    index for the player. The index is in the range 0 up to the number of
    players minus 1.
    """
    player_indices = {}
    player_index = 0
    players = set()
    for i, j, _ in game_results:
        players.add(i)
        players.add(j)

    for player in sorted(players):
        player_indices[player] = player_index
        player_index += 1

    return player_indices


def compute_log_likelihood(wins, gamma):
    """Computes the log likelihood of the game results given gamma.

    The model is P(i beats j) = gamma[i] / (gamma[i] + gamma[j]).

    Parameters
    ----------
    wins: ndarray
        An N x N matrix with ij entry equal to the number of times that i
        beat j. Assumes diagonal is zero.
    gamma: ndarray
        A shape (N,) array whose ith entry is player i's rating.
    """
    # Create a matrix with (i, j) entry gamma[i] + gamma[j].
    gamma_sum = gamma[:, np.newaxis] + gamma[:, np.newaxis].T

    # Transform gamma into an (N, 1) array
    gamma_rows = gamma[:, np.newaxis]

    return np.sum(wins * (np.log(gamma_rows) - np.log(gamma_sum)))


def compute_win_matrix(game_results, player_indices):
    """Computes the win matrix for the game results and player indices. The
    ij entry is the number of times i beat j.
    """
    players = set(i for i, _, _ in game_results)
    players.union(set(j for _, j, _ in game_results))
    num_players = len(players)
    wins = np.zeros(num_players, num_players)
    for i, j, n in game_results:
        wins[player_indices[i], player_indices[j]] += n

    return wins


def elo(game_results: Sequence[Tuple], reference_gammas: Dict[int, float] =
    None):
    """Computes the elo ratings for players given some game results.

    Uses the model:
        P(i beats j) = gamma[i] / (gamma[i] + gamma[j]).

    Parameters
    ----------
    game_results:
        A sequence where the elements are tuples (i, j, n). This means i beat j
        n times.
    reference_gammas:
        Fix some of the gammas as reference values.

    Results
    -------
    gamma: dict
        Dictionary with keys the players and values their elo ratings.
    """
    player_indices = compute_player_indices(game_results)
    wins = compute_win_matrix(game_results, player_indices)

    # Initialise gamma randomly
    gamma = np.random.rand(len(player_indices))

    # Fix reference values, if given
    reference_gammas_v = np.zeros(len(player_indices))
    if reference_gammas:
        for i, g in reference_gammas:
            reference_gammas_v[player_indices[i]] = g
    gamma = np.where(reference_gammas_v > 0, reference_gammas_v, gamma)

    return run_mm(gamma, wins, reference_gammas=reference_gammas_v)


def update_gamma(gamma, wins):
    """Updates gamma by one step.
    """
    # Create a matrix with (i, j) entry gamma[i] + gamma[j].
    gamma_sum = gamma[:, np.newaxis] + gamma[:, np.newaxis].T

    # Pairings has (i, j) entry equal to the number of pairings between
    # i and j. That is, number of times i beats j plus number of times j
    # beats i.
    pairings = wins + wins.T

    gamma = np.sum(wins, axis=1) / np.sum(pairings / gamma_sum, axis=1)
    return gamma.ravel()


def run_mm(initial_gamma, wins, num_iters=30, reference_gammas: np.ndarray =
    None):
    """Runs minorisation maximisation (Hunter).

    Optionally use reference_gammas to fix some of the gamma values. Then
    this algorithm computes the maximum likelihood gamma values with these
    reference gammas fixed.

    Parameters
    ----------
    initial_gamma: ndarray
        The initial value of gamma to use. This should have all positive
        entries.
    wins: ndarray
        An N x N matrix, where there are N players.
    num_iters: int
        The number of iterations to run the algorithm for.
    reference_gammas:
        A numpy array with ith entry either 0, if no reference gamma,
        or a positive float with the fixed value of gamma for the ith player.
    """
    gamma = initial_gamma / np.sum(initial_gamma)
    assert np.all(gamma > 0)

    for it in range(num_iters):
        # Update gamma
        gamma = update_gamma(gamma, wins)

        # Fix reference values, if given
        if reference_gammas is not None:
            gamma = np.where(reference_gammas > 0, reference_gammas, gamma)

        log_likelihood = compute_log_likelihood(wins, gamma)
        print("Log likelihood: {}".format(log_likelihood))

    return gamma
