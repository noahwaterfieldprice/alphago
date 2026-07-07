import math

import numpy as np

from alphago.elo import (
    compute_log_likelihood,
    compute_player_indices,
    compute_win_matrix,
    elo,
    run_mm,
    update_gamma,
)


def test_compute_log_likelihood():
    gamma = np.array([1, 2, 3])
    wins = np.array([[0, 3, 4], [1, 0, 2], [2, 0, 0]])

    expected = (
        3 * np.log(1 / (1 + 2))
        + 4 * np.log(1 / (1 + 3))
        + 1 * np.log(2 / (2 + 1))
        + 2 * np.log(2 / (2 + 3))
        + 2 * np.log(3 / (3 + 1))
    )

    computed = compute_log_likelihood(wins, gamma)
    assert expected == computed


def test_compute_win_matrix_includes_losers_only_players():
    # Player 2 appears only as the loser j, never as the winner i.
    game_results = [(1, 2, 5)]
    player_indices = compute_player_indices(game_results)

    wins = compute_win_matrix(game_results, player_indices)

    assert wins.shape == (2, 2)


def test_run_mm():
    initial_gamma = np.array([1, 1, 1])
    wins = np.array([[0, 30, 40], [1, 0, 20], [2, 0, 0]])

    initial_ll = compute_log_likelihood(wins, initial_gamma)
    gamma = run_mm(initial_gamma, wins)
    final_ll = compute_log_likelihood(wins, gamma)
    assert initial_ll < final_ll


def test_run_mm_large():
    # Generate fake data according to the Bradley-Terry model.
    hidden_gamma = np.array([10, 20, 1, 1, 5, 3, 100, 8, 100, 10])
    hidden_gamma = hidden_gamma / np.sum(hidden_gamma)
    wins = np.zeros((10, 10))
    num_games = 1000
    for i in range(10):
        for j in range(10):
            if i == j:
                continue
            wins[i, j] = int(
                num_games * hidden_gamma[i] / (hidden_gamma[i] + hidden_gamma[j])
            )

    hidden_ll = compute_log_likelihood(wins, hidden_gamma)

    initial_gamma = np.random.rand(10)
    initial_gamma = initial_gamma / np.sum(initial_gamma)
    initial_ll = compute_log_likelihood(wins, initial_gamma)
    gamma = run_mm(initial_gamma, wins)
    final_ll = compute_log_likelihood(wins, gamma)
    print(f"Hidden ll: {hidden_ll}, initial ll: {initial_ll}, final ll: {final_ll}")
    print(hidden_gamma * 100)
    print(gamma * 100)
    assert initial_ll < final_ll


def test_update_gamma():
    initial_gamma = np.array([1, 2, 3])
    wins = np.array([[0, 3, 4], [1, 0, 2], [2, 0, 0]])

    expected0 = 7 / (4 / 3 + 6 / 4)
    expected1 = 3 / (4 / 3 + 2 / 5)
    expected2 = 2 / (6 / 4 + 2 / 5)

    expected = np.array([expected0, expected1, expected2])
    computed = update_gamma(initial_gamma, wins)
    assert (expected == computed).all()


def test_reference_gammas():
    initial_gamma = np.array([1, 1, 1])
    wins = np.array([[0, 30, 40], [1, 0, 20], [2, 0, 0]])

    initial_ll = compute_log_likelihood(wins, initial_gamma)

    reference_gammas = np.array([0, 17, 0])
    gamma = run_mm(initial_gamma, wins, reference_gammas=reference_gammas)
    final_ll = compute_log_likelihood(wins, gamma)
    assert initial_ll < final_ll

    assert gamma[1] == reference_gammas[1]


def test_elo_rates_dominant_player_higher():
    # Player 1 dominates player 2; the third player keeps the win graph
    # connected so the MM algorithm has a well-defined solution. Player 1
    # never loses to player 2, so its rating must strictly exceed player 2's.
    # Before the fix, elo() returned random gammas and this ordering
    # only held by chance.
    game_results = [(1, 2, 10), (2, 3, 5), (3, 1, 1)]

    ratings = elo(game_results)

    assert ratings[1] > ratings[2]


def test_elo_finite_when_player_has_zero_wins():
    # Player 3 never wins a single game, which drives update_gamma's
    # sum-of-wins numerator to zero. Before the fix, the unguarded
    # `pairings / gamma_sum` division produced a 0/0 = NaN that cascaded to
    # every rating, returning {1: nan, 2: nan, 3: nan}. The guarded division
    # and gamma floor must keep every rating finite and correctly ordered.
    game_results = [(1, 2, 10), (2, 3, 5), (1, 3, 4)]

    ratings = elo(game_results)

    assert all(math.isfinite(v) for v in ratings.values())
    # Player 3 lost every game, so it must be the strictly lowest rating.
    assert ratings[3] == min(ratings.values())
    assert ratings[3] < ratings[1]
    assert ratings[3] < ratings[2]


def test_elo_reference_gammas_dict_does_not_crash():
    # elo() must accept reference_gammas as a dict keyed by player number.
    # Before the fix, `for i, g in reference_gammas` iterated the dict
    # keys and raised a TypeError while trying to unpack an int.
    game_results = [(1, 2, 10), (2, 3, 5), (3, 1, 1)]

    ratings = elo(game_results, reference_gammas={1: 2.0})

    assert set(ratings.keys()) == {1, 2, 3}
    assert ratings[1] == 2.0
