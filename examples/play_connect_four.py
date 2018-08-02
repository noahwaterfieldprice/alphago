"""This program plays connect four using Monte Carlo Tree Search and a trivial
estimator. For nonterminal states, the evaluator returns the uniform
probability distribution over available actions and a value of 0. In a
terminal state, we back up the utility returned by the game.
"""

import argparse

import numpy as np

from alphago.games.connect_four import ConnectFour
from alphago.estimator import ConnectFourNet
from alphago import mcts_tree
from alphago.estimator import create_trivial_estimator


def load_net(checkpoint):
    """Load the network at the checkpoint.

    Parameters
    ----------
    checkpoint: str
        Should be the name of the checkpoint, including the path. Probably
        ends in '.checkpoint'.

    Returns
    -------
    AbstractNeuralNetEstimator
        The network loaded.
    """
    game = ConnectFour()
    estimator = ConnectFourNet(learning_rate=1e-4,
                               l2_weight=1e-4, value_weight=0.01,
                               action_indices=game.action_indices)
    estimator.restore(checkpoint)
    return estimator


def play_game(human, estimator, mcts_iters, c_puct):
    """

    Parameters
    ----------
    human: int
        1 if human plays first, otherwise 2 and human plays second.

    Returns
    -------
    int
        Result of the game: 1 if you win, 0 if you draw, -1 if you lose.
    """
    cf = ConnectFour()
    state = cf.initial_state
    computer = 1 if human == 2 else 2
    print("You are player: {}".format(human))
    while not cf.is_terminal(state):
        player = cf.current_player(state)
        next_states = cf.legal_actions(state)
        if player == computer:
            root = mcts_tree.MCTSNode(state, player)
            if mcts_iters == 0:
                # Choose the maximum probability action of the net.
                action_probs, _ = estimator(state)
                action = max(action_probs, key=action_probs.get)
            else:
                action_probs = mcts_tree.mcts(
                    root, cf, estimator, mcts_iters, c_puct)
                actions, probs = zip(*action_probs.items())
                print("Action probabilities: {}".format(action_probs))
                action_ix = np.random.choice(range(len(actions)), p=probs)
                action = actions[action_ix]
            print("Taking action: {}".format(action))
        else:
            action = None
            while action not in next_states:
                action_ix = int(input("Your move (1-7 reading "
                                      "across the board): "))
                action_ix -= 1
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
    return utility[human]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--player', help='1 if you play first, 2 if you play '
                                         'second.')
    parser.add_argument('--checkpoint',
                        help='The checkpoint path to use for the estimator. '
                             'If not given, then use a trivial estimator.')
    parser.add_argument('--mcts_iters', help='If 0, then just use the raw '
                                             'network.')

    args = parser.parse_args()

    mcts_iters = int(args.mcts_iters) if args.mcts_iters is not None else 1000
    c_puct = 0.5

    if args.player:
        human = int(args.player)
    else:
        human = np.random.choice([1, 2])

    if args.checkpoint:
        estimator = load_net(args.checkpoint)
    else:
        cf = ConnectFour()
        estimator = create_trivial_estimator(cf)

    play_game(human, estimator, mcts_iters, c_puct)
