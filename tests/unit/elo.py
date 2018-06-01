import numpy as np

from alphago.elo import compute_log_likelihood, run_mm, update_gamma


def test_compute_log_likelihood():
    gamma = np.array([1, 2, 3])
    wins = np.array([[0, 3, 4],
                     [1, 0, 2],
                     [2, 0, 0]])

    expected = 3 * np.log(1/(1+2)) + 4 * np.log(1/(1+3)) + \
        1 * np.log(2/(2+1)) + 2 * np.log(2/(2+3)) + 2 * np.log(3/(3+1))

    computed = compute_log_likelihood(wins, gamma)
    assert expected == computed


def test_run_mm():
    initial_gamma = np.array([1, 1, 1])
    wins = np.array([[0, 30, 40],
                     [1, 0, 20],
                     [2, 0, 0]])

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
            wins[i, j] = int(num_games * hidden_gamma[i] / (hidden_gamma[i] +
                                                            hidden_gamma[j]))

    hidden_ll = compute_log_likelihood(wins, hidden_gamma)

    initial_gamma = np.random.rand(10)
    initial_gamma = initial_gamma / np.sum(initial_gamma)
    initial_ll = compute_log_likelihood(wins, initial_gamma)
    gamma = run_mm(initial_gamma, wins)
    final_ll = compute_log_likelihood(wins, gamma)
    print("Hidden ll: {}, initial ll: {}, final ll: {}".format(hidden_ll,
                                                               initial_ll,
                                                               final_ll))
    print(hidden_gamma * 100)
    print(gamma * 100)
    assert initial_ll < final_ll


def test_update_gamma():
    initial_gamma = np.array([1, 2, 3])
    wins = np.array([[0, 3, 4],
                     [1, 0, 2],
                     [2, 0, 0]])

    expected0 = 7 / (4 / 3 + 6 / 4)
    expected1 = 3 / (4 / 3 + 2 / 5)
    expected2 = 2 / (6 / 4 + 2 / 5)

    expected = np.array([expected0, expected1, expected2])
    computed = update_gamma(initial_gamma, wins)
    assert (expected == computed).all()


def test_reference_gammas():
    initial_gamma = np.array([1, 1, 1])
    wins = np.array([[0, 30, 40],
                     [1, 0, 20],
                     [2, 0, 0]])

    initial_ll = compute_log_likelihood(wins, initial_gamma)

    reference_gammas = np.array([0, 17, 0])
    gamma = run_mm(initial_gamma, wins, reference_gammas=reference_gammas)
    final_ll = compute_log_likelihood(wins, gamma)
    assert initial_ll < final_ll

    assert gamma[1] == reference_gammas[1]
