INITIAL_STATE = 0
TERMINAL_STATE_VALUES = (1,) * 12
PLAYER1_STATES = [0, 3, 4, 5, 6]
PLAYER2_STATES = [1, 2]
WHICH_PLAYER_MAP = {state: 1 for state in PLAYER1_STATES}
WHICH_PLAYER_MAP.update({state: 2 for state in PLAYER2_STATES})
NEXT_STATES_MAP = {
        0: (1, 2),
        1: (3, 4), 2: (5, 6),
        3: (7, 8, 9), 4: (10, 11, 12), 5: (13, 14, 15), 6: (16, 17, 18)
}


def which_player(state):
    if is_terminal(state):
        return None
    return WHICH_PLAYER_MAP[state]


def is_terminal(state):
    return state >= 7


def compute_next_states(state):
    if is_terminal(state):
        raise ValueError("Next states can not be generated for a "
                         "terminal state.")

    return {action: next_state for action, next_state
            in zip(range(3), NEXT_STATES_MAP[state])}


def utility(state):
    if not is_terminal(state):
        raise ValueError("Utility can not be calculated for non-terminal "
                         "state.")

    player1_value = TERMINAL_STATE_VALUES[state - 7]
    return {1: player1_value, 2: -player1_value}


def mock_evaluator(state):
    prior_probs = {action: 1 / 3 for action in range(3)}

    value = 0.0
    return prior_probs, value
