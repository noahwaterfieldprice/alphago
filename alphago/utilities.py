import numpy as np


def compute_distribution(d):
    """Calculate a probability distribution with probabilities
    proportional to the values in a dictionary

    Parameters
    ----------
    d: dict
        A dictionary with values equal to positive floats.

    Returns
    -------
    prob_distribution: dict:
        A probability distribution proportional to the values of d,
        given as a dictionary with keys equal to those of d and values
        the probability corresponding to the value.
    """
    total = sum(d.values())
    assert min(d.values()) >= 0
    assert total > 0
    prob_distribution = {k: float(v) / float(total)
                         for k, v in d.items()}
    return prob_distribution


def sample_distribution(distribution):
    # TODO: write docstring and UTs
    """

    Parameters
    ----------
    distribution: dict
        A dictionary with keys the outcomes and values the probabilities.
    """

    outcomes, probabilities = zip(*distribution.items())
    outcome_ix = np.random.choice(len(outcomes), p=probabilities)
    action = outcomes[outcome_ix]

    return action
