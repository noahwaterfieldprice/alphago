import random

from alphago.games.connect_four import action_list_to_state, ConnectFour



def generate_states(n, min_length=1):
    """Generates n random connect four states, uniformly in length, and then
    uniformly across states of that length.

    Parameters
    ----------
    n: int
        The number of states to generate.

    Returns
    -------
    states: set
        A set of states.
    """
    states = set()
    game = ConnectFour()
    while len(states) <= n:
        cols = [i for i in range(7)] * 6
        random.shuffle(cols)
        l = random.randint(min_length, 43)
        state = cols[:l]
        state_array = action_list_to_state(state)

        if not game.is_terminal(state_array):
            states.add(tuple(x + 1 for x in state))

    return states


if __name__ == "__main__":
    states = generate_states(100000, 8)

    with open('output.txt', 'w') as f:
        string_states = [''.join((str(x) for x in state)) for state in states]
        f.write('\n'.join(string_states))
