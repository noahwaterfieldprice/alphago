
class AbstractState:
    
    initial_state = None

    def vector(state):
        raise NotImplementedError

    def available_actions(state):
        raise NotImplementedError

    def next_state(state, action):
        raise NotImplementedError

    def is_terminal(state):
        raise NotImplementedError

    def utility(state):
        raise NotImplementedError

