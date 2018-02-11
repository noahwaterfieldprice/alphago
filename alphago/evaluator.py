import numpy as np

from alphago.mcts_tree import compute_distribution


def trivial_evaluator(state, next_states_function, action_space, is_terminal,
                      utility, which_player):
    """Evaluates a game state for a game. It is trivial in the sense that it
    returns the uniform probability distribution over all actions in the game.

    Parameters
    ----------
    state: tuple
        A state in the game.
    next_states_function: func
        Returns a dictionary from actions available in the current state to the
        resulting game states.
    action_space: list
        A list of all actions in the game.
    is_terminal: func
        Takes a game state and returns whether or not it is terminal.
    utility: func
        Given a terminal game state, returns an Outcome
    which_player: func
        Given a state, return whether it is player 1 or player 2 to play.

    Returns
    -------
    probs: dict
        A dictionary from actions to probabilities. Some actions might not be
        legal in this game state, but the evaluator returns a probability for
        choosing each one.
    value: float
        The evaluator's estimate of the value of the state 'state'.
    """

    if is_terminal(state):
        value = utility(state)
        value = value.player1 if which_player(state) == 1 else value.player2
        probs = {}
        return probs, value

    next_states = next_states_function(state)

    return compute_distribution({a: 1.0 for a in next_states}), 0.0
