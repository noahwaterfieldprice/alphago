"""This program plays noughts and crosses using Monte Carlo Tree Search and a
trivial evaluator. For nonterminal states, the evaluator returns the uniform
probability distribution over available actions and a value of 0. In a terminal
state, we back up the utility returned by the game.
"""

from dataclasses import dataclass

import numpy as np

from alphago.config import load_config
from alphago.estimator import create_trivial_estimator
from alphago.games.noughts_and_crosses import NoughtsAndCrosses
from alphago.player import MCTSPlayer


@dataclass
class PlayNoughtsAndCrossesConfig:
    """Interactive noughts and crosses play knobs (dotlist overrides).

    Attributes:
        player: 1 to play first, 2 to play second; ``None`` picks at random.
        mcts_iters: Simulations per computer move.
        c_puct: PUCT exploration constant.
        tau: Sampling temperature; closer to 0 exploits more.
    """

    player: int | None = None
    mcts_iters: int = 2000
    c_puct: float = 0.5
    tau: float = 0.01


def main(cfg) -> None:
    """Run an interactive noughts and crosses game driven by ``cfg``."""
    nac = NoughtsAndCrosses()
    evaluator = create_trivial_estimator(nac)

    state = nac.initial_state
    if cfg.player is not None:
        human_player_no = int(cfg.player)
        computer_player_no = 1 if human_player_no == 2 else 2
    else:
        computer_player_no = np.random.choice([1, 2])
        human_player_no = 1 if computer_player_no == 2 else 2
    computer_player = MCTSPlayer(
        nac, evaluator, mcts_iters=cfg.mcts_iters, c_puct=cfg.c_puct, tau=cfg.tau
    )
    print(f"You are player: {human_player_no}")
    while not nac.is_terminal(state):
        player_no = nac.current_player(state)
        next_states = nac.legal_actions(state)
        if player_no == computer_player_no:
            action = computer_player.choose_action(state)
            computer_player.update(action)
            print(f"Taking action: {action}")
        else:
            action = None
            while action not in next_states:
                action_ix = int(input("Your move (0-8 reading across the board): "))
                if 0 <= action_ix <= 8:
                    action = nac.action_space[action_ix]
                    computer_player.update(action)
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


if __name__ == "__main__":
    main(load_config(PlayNoughtsAndCrossesConfig))
