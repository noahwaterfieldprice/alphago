import numpy as np

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

class NoughtsAndCrossesState(AbstractState):

    initial_state = (np.nan, ) * 9
    
    def is_terminal(state):
        """ Given a state, returns whether it is terminal. The state is a tuple
        of shape (9,) with -1 for 'O', 1 for 'X' and np.nan for empty square.
        """
        arr = np.array(state)
        arr = arr.reshape(3,3)
        print("State: {}".format(state))
        print("Array: {}".format(arr))
        if np.any(np.abs(np.nansum(arr, axis=0))==3):
            return True
        if np.any(np.abs(np.nansum(arr, axis=1))==3):
            return True
        if abs(np.nansum(arr.diagonal())) == 3:
            return True
        minor = arr[2][0] + arr[1][1] + arr[0][2]
        if abs(minor) == 3:
            return True
        return False
