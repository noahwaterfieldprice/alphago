import numpy as np

from .utilities import memoize


def solve_game_alpha_beta(game, state, alpha, beta, depth, heuristic=None):
    if game.is_terminal(state):
        return game.utility(state)[1], None

    player = game.which_player(state)

    if heuristic is not None and depth == 0:
        # The heuristic is for player 1, so we negate it if it's player 2's
        # turn.
        heuristic_value = heuristic(state) if player == 1 else -heuristic(state)
        return heuristic_value, None
    # Always use player 1's values. Player 1 wants to maximise, player 2 wants
    # to minimise.
    if player == 1:
        next_states = game.compute_next_states(state)
        v = -np.inf
        best_action = None
        for action, child_state in next_states.items():
            child_utility, _ = solve_game_alpha_beta(
                game, child_state, alpha, beta, depth-1, heuristic=heuristic)
            if best_action is None or child_utility > v:
                best_action = action
            v = max(v, child_utility)
            alpha = max(alpha, v)
            if beta <= alpha:
                break
        return v, best_action
    else:
        # We are player 2 -- try to minimise player 1's value.
        next_states = game.compute_next_states(state)
        v = np.inf
        best_action = None
        for action, child_state in next_states.items():
            child_utility, _ = solve_game_alpha_beta(
                game, child_state, alpha, beta, depth-1, heuristic=heuristic)
            if best_action is None or child_utility < v:
                best_action = action
            v = min(v, child_utility)
            beta = min(beta, v)
            if beta <= alpha:
                break
        return v, best_action


def solve_game(best_actions, game, state):
    """Returns the value of the state (to both players) and the optimal move to
    play in this state. Fills in the dictionary 'best_actions' with the best
    actions to take in all the states that are descendants of state (including
    state).
    """
    if game.is_terminal(state):
        return game.utility(state), None
    else:
        # If the state is not terminal, then the player plays the move that
        # maximises their utility
        player = game.which_player(state)
        next_states = game.compute_next_states(state)

        child_utilities = {}
        for action, child_state in next_states.items():
            child_utility, _ = solve_game(
                best_actions, game, child_state)
            child_utilities[action] = child_utility
        best_action = max(child_utilities.keys(), key=lambda x:
                          child_utilities[x][player])

        best_actions[state] = best_action
        return child_utilities[best_action], best_action


@memoize
def backwards_induction(game, state):
    """Returns the value of the state (to both players) and the optimal move to
    play in this state.
    """
    best_actions = {}
    value, best_action = solve_game(best_actions, game, state)
    return value, best_action
