"""This program plays noughts and crosses using Monte Carlo Tree Search and a
trivial evaluator. For nonterminal states, the evaluator returns the uniform
probability distribution over available actions and a value of 0. In a terminal
state, we back up the utility returned by the game.
"""
import numpy as np

import alphago.games.noughts_and_crosses as nac
from alphago import mcts_tree
from alphago.estimator import create_trivial_estimator
from alphago.player import MCTSPlayer
if __name__ == "__main__":

    evaluator = create_trivial_estimator(nac.compute_next_states)

    state = nac.INITIAL_STATE
    computer_player_no = np.random.choice([1, 2])
    computer_player = MCTSPlayer(computer_player_no, nac, evaluator,
                                 mcts_iters=2000, c_puct=0.5)
    human_player_no = 1 if computer_player_no == 2 else 2
    print("You are player: {}".format(human_player_no))
    while not nac.is_terminal(state):
        player_no = nac.which_player(state)
        next_states = nac.compute_next_states(state)
        if player_no == computer_player_no:
            current_node = mcts_tree.MCTSNode(state, player_no)
            action_probs = computer_player.action_probabilities(
                current_node.game_state)

            actions, probabilities = zip(*action_probs.items())
            print("Action probabilities: {}".format(action_probs))
            action_ix = np.random.choice(range(len(actions)), p=probabilities)
            action = actions[action_ix]
            print("Taking action: {}".format(action))
        else:
            action = None
            while action not in next_states:
                action_ix = int(input("Your move (0-8 reading "
                                      "across the board): "))
                if 0 <= action_ix <= 8:
                    action = nac.ACTION_SPACE[action_ix]
        state = next_states[action]

        nac.display(state)
        print("\n")

    # The state is terminal, so let's see who won.
    utility = nac.utility(state)
    if utility[human_player_no] == 1:
        print("You win!")
    elif utility[human_player_no] == 0:
        print("It's a draw!")
    else:
        print("You lose!")
