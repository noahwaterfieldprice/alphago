"""This program plays connect four using Monte Carlo Tree Search and a trivial
estimator. For nonterminal states, the evaluator returns the uniform
probability distribution over available actions and a value of 0. In a
terminal state, we back up the utility returned by the game.
"""

from dataclasses import dataclass

import numpy as np

from alphago import mcts_tree
from alphago.config import load_config
from alphago.estimator import ConnectFourNet, create_trivial_estimator
from alphago.games.connect_four import ConnectFour, optimal_moves


@dataclass
class PlayConnectFourConfig:
    """Interactive Connect Four play knobs (overridable via dotlist).

    Attributes:
        player: 1 to play first, 2 to play second; ``None`` picks at random.
        checkpoint: Estimator checkpoint path; ``None`` uses a trivial estimator.
        mcts_iters: Simulations per computer move; ``0`` uses the raw network.
        tau: Sampling temperature; closer to 0 exploits more.
        c_puct: PUCT exploration constant.
    """

    player: int | None = None
    checkpoint: str | None = None
    mcts_iters: int = 1000
    tau: float = 1.0
    c_puct: float = 0.5


def load_net(checkpoint):
    """Load the network at the checkpoint.

    Parameters
    ----------
    checkpoint: str
        Should be the name of the checkpoint, including the path. Probably
        ends in '.pt'. Loaded via the estimator's ``restore`` (``torch.load``
        with ``map_location``).

    Returns
    -------
    AbstractNeuralNetEstimator
        The network loaded.
    """
    game = ConnectFour()
    estimator = ConnectFourNet(
        learning_rate=1e-4,
        l2_weight=1e-4,
        value_weight=0.01,
        action_indices=game.action_indices,
    )
    estimator.restore(checkpoint)
    return estimator


def play_game(human, estimator, mcts_iters, c_puct, tau):
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
    print(f"You are player: {human}")
    action_list = []
    while not cf.is_terminal(state):
        player = cf.current_player(state)
        next_states = cf.legal_actions(state)
        if player == computer:
            root = mcts_tree.MCTSNode(state, player)
            if mcts_iters == 0:
                # Choose the maximum probability action of the net. The
                # estimator returns unmasked probabilities over ALL actions,
                # so restrict the argmax to legal columns;
                # otherwise a full column could be picked and `next_states`
                # lookup below would raise KeyError.
                action_probs, _ = estimator(state)
                action = max(next_states, key=lambda a: action_probs[a])
            else:
                action_probs = mcts_tree.mcts(
                    root, cf, estimator, mcts_iters=mcts_iters, c_puct=c_puct, tau=tau
                )
                actions, probs = zip(*action_probs.items(), strict=False)
                print(f"Action probabilities: {action_probs}")
                action_ix = np.random.choice(range(len(actions)), p=probs)
                action = actions[action_ix]
            print(f"Taking action: {action + 1}")
        else:
            action = None
            while action not in next_states:
                user_input = input("Your move (1-7 reading across the board): ")
                if user_input == "cheat":
                    print(f"Optimal moves: {optimal_moves(action_list)}")
                    continue
                action_ix = int(user_input)
                action_ix -= 1
                if 0 <= action_ix < len(cf.action_space):
                    action = cf.action_space[action_ix]
        action_list.append(action + 1)
        state = next_states[action]

        cf.display(state)
        print("\n")
        print("Action list: {}".format("".join(map(str, action_list))))
    # The state is terminal, so let's see who won.
    utility = cf.utility(state)
    if utility[human] == 1:
        print("You win!")
    elif utility[human] == 0:
        print("It's a draw!")
    else:
        print("You lose!")
    return utility[human]


def main(cfg) -> None:
    """Run an interactive Connect Four game driven by ``cfg``."""
    human = int(cfg.player) if cfg.player is not None else np.random.choice([1, 2])

    if cfg.checkpoint:
        estimator = load_net(cfg.checkpoint)
    else:
        cf = ConnectFour()
        estimator = create_trivial_estimator(cf)

    play_game(human, estimator, cfg.mcts_iters, cfg.c_puct, cfg.tau)


if __name__ == "__main__":
    main(load_config(PlayConnectFourConfig))
