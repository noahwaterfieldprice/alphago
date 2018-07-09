"""This program plays connect four using Monte Carlo Tree Search and a trivial
evaluator. For nonterminal states, the evaluator returns the uniform probability
distribution over available actions and a value of 0. In a terminal state, we
back up the utility returned by the game.
"""


import numpy as np

from alphago.games.connect_four import ConnectFour
from alphago import mcts_tree
from alphago.estimator import create_trivial_estimator

if __name__ == "__main__":

    max_iters = 1000
    c_puct = 0.5

    cf = ConnectFour()
    evaluator = create_trivial_estimator(cf.legal_actions)

    state = cf.initial_state
    computer = np.random.choice([1, 2])
    human = 1 if computer == 2 else 2
    print("You are player: {}".format(human))
    while not cf.is_terminal(state):
        player = cf.current_player(state)
        next_states = cf.legal_actions(state)
        if player == computer:
            root = mcts_tree.MCTSNode(state, player)
            action_probs = mcts_tree.mcts(
                root, evaluator, cf,  max_iters, c_puct)
            actions, probs = zip(*action_probs.items())
            print("Action probabilities: {}".format(action_probs))
            action_ix = np.random.choice(range(len(actions)), p=probs)
            action = actions[action_ix]
            print("Taking action: {}".format(action))
        else:
            action = None
            while action not in next_states:
                action_ix = int(input("Your move (0-6 reading "
                                      "across the board): "))
                if 0 <= action_ix < len(cf.action_space):
                    action = cf.action_space[action_ix]
        state = next_states[action]

        cf.display(state)
        print("\n")

    # The state is terminal, so let's see who won.
    utility = cf.utility(state)
    if utility[human] == 1:
        print("You win!")
    elif utility[human] == 0:
        print("It's a draw!")
    else:
        print("You lose!")
