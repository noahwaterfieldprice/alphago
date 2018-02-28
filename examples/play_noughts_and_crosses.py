import numpy as np

import alphago.games.noughts_and_crosses as nac
from alphago import mcts_tree
from alphago.evaluator import create_trivial_evaluator

if __name__ == "__main__":

    max_iters = 5000
    c_puct = 0.5

    action_space = [(i, j) for i in range(3) for j in range(3)]

    evaluator = create_trivial_evaluator(nac.compute_next_states)

    state = nac.INITIAL_STATE
    computer = np.random.choice([1, 2])
    human = 1 if computer == 2 else 2
    print("You are player: {}".format(human))
    while not nac.is_terminal(state):
        player = nac.which_player(state)
        if player == computer:
            action_probs = mcts_tree.mcts(
                mcts_tree.MCTSNode(state, player), evaluator,
                nac.compute_next_states, nac.utility, nac.which_player,
                nac.is_terminal, max_iters, c_puct)
            actions, probs = zip(*action_probs.items())
            print(action_probs)
            action_ix = np.random.choice(range(len(actions)), p=probs)
            action = actions[action_ix]
            print("Taking action: {}".format(action))
        else:
            action = int(input("Your move (1-9 reading "
                               "across the board): "))
            assert (action >= 1) and (action <= 9)
            action = action_space[action-1]
        next_states = nac.compute_next_states(state)
        state = next_states[action]

        nac.display(state)
        print("\n")

    # The state is terminal, so let's see who won.
    utility = nac.utility(state)
    if utility[human] == 1:
        print("You win!")
    elif utility[human] == 0:
        print("It's a draw!")
    else:
        print("You lose!")
