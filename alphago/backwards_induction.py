from .utilities import memoize


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
