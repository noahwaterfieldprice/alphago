import inspect
from typing import Any, Callable, TypeVar, Dict

import numpy as np

T = TypeVar('T')


def sample_distribution(distribution: Dict[T, float]) -> T:
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


def memoize_instance(instance: T) -> None:
    """Given an instance of a class, replace each of its methods with
    a memoized copy."""
    for name, fn in inspect.getmembers(instance, inspect.ismethod):
        setattr(instance, name, memoize(fn))
