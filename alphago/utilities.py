import inspect

import numpy as np


def sample_distribution(distribution):
    """Given a probability distribution as a dictionary, with keys the
    outcomes and values the probabilities, sample an outcome from the
    distribution according to the probabilities.

    Parameters
    ----------
    distribution: dict
        A dictionary with keys the outcomes and values the probabilities.
    """

    outcomes, probabilities = zip(*distribution.items())
    outcome_ix = np.random.choice(len(outcomes), p=probabilities)
    outcome = outcomes[outcome_ix]

    return outcome


def memoize(func):
    cache = dict()

    def memoized_func(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result

    return memoized_func


def memoize_instance(instance):
    for name, fn in inspect.getmembers(instance, inspect.ismethod):
        setattr(instance, name, memoize(fn))
