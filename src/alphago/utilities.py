import inspect
from collections.abc import Callable
from typing import Any

import numpy as np


def sample_distribution[T](distribution: dict[T, float]) -> T:
    """Given a probability distribution as a dictionary, with keys the
    outcomes and values the probabilities, sample an outcome from the
    distribution according to the probabilities.

    Parameters
    ----------
    distribution: dict
        A dictionary with keys the outcomes and values the probabilities.
    """

    outcomes, probabilities = zip(*distribution.items(), strict=False)
    outcome_ix = np.random.choice(len(outcomes), p=probabilities)
    outcome = outcomes[outcome_ix]

    return outcome


def memoize(func: Callable) -> Callable:
    """Given a functon, return a memoized copy of that function."""
    cache = dict()

    def memoized_func(*args: Any) -> Any:
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result

    return memoized_func


def memoize_instance[T](instance: T) -> None:
    """Given an instance of a class, replace each of its methods with
    a memoized copy."""
    for name, fn in inspect.getmembers(instance, inspect.ismethod):
        setattr(instance, name, memoize(fn))
