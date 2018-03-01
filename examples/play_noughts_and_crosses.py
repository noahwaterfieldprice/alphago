"""This program plays noughts and crosses using Monte Carlo Tree Search and a
trivial evaluator. For nonterminal states, the evaluator returns the uniform
probability distribution over available actions and a value of 0. In a terminal
state, we back up the utility returned by the game.
"""


import numpy as np

import alphago.games.noughts_and_crosses as nac
from alphago import mcts_tree
from alphago.evaluator import create_trivial_evaluator

if __name__ == "__main__":

    max_iters = 5000
    c_puct = 0.5

    evaluator = create_trivial_evaluator(nac.compute_next_states)

    state = nac.INITIAL_STATE
    computer = np.random.choice([1, 2])
    human = 1 if computer == 2 else 2
    print("You are player: {}".format(human))
    while not nac.is_terminal(state):
        player = nac.which_player(state)
        next_states = nac.compute_next_states(state)
        if player == computer:
            root = mcts_tree.MCTSNode(state, player)
            action_probs = mcts_tree.mcts(
                root, evaluator,
                nac.compute_next_states, nac.utility, nac.which_player,
                nac.is_terminal, max_iters, c_puct)
            mcts_tree.print_tree(root)
            actions, probs = zip(*action_probs.items())
            print("Action probabilities: {}".format(action_probs))
            action_ix = np.random.choice(range(len(actions)), p=probs)
            action = actions[action_ix]
            print("Taking action: {}".format(action))
        else:
            action = None
            while action not in next_states:
                action_ix = int(input("Your move (0-8 reading "
                                      "across the board): "))
                if 0 <= action_ix and action_ix <= 8:
                    action = nac.ACTION_INDICES[action_ix]
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
