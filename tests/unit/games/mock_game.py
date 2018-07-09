class MockGame:

    def __init__(self):
        self.initial_state = 0
        self.terminal_state_values = (1,) * 12
        self.player1_states = [0, 3, 4, 5, 6]
        self.player2_states = [1, 2]
        self.current_player_map = {state: 1 for state in self.player1_states}
        self.current_player_map.update({state: 2 for state in self.player2_states})
        self.legal_actions_map = {
            0: (1, 2),
            1: (3, 4), 2: (5, 6),
            3: (7, 8, 9), 4: (10, 11, 12), 5: (13, 14, 15), 6: (16, 17, 18)
        }

    def current_player(self, state):
        if self.is_terminal(state):
            return None
        return self.current_player_map[state]

    def is_terminal(self, state):
        return state >= 7

    def legal_actions(self, state):
        if self.is_terminal(state):
            raise ValueError("Next states can not be generated for a "
                             "terminal state.")

        return {action: next_state for action, next_state
                in zip(range(3), self.legal_actions_map[state])}

    def utility(self, state):
        if not self.is_terminal(state):
            raise ValueError("Utility can not be calculated for non-terminal "
                             "state.")

        player1_value = self.terminal_state_values[state - 7]
        return {1: player1_value, 2: -player1_value}

    def mock_estimator(self, state):
        prior_probs = {action: 1 / 3 for action in range(3)}

        value = 0.0
        return prior_probs, value

    def __repr__(self):
        return "MockGame"
