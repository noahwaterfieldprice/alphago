import alphago.games as games
from alphago.estimator import create_trivial_estimator
from alphago.evaluator import evaluate
from alphago.player import MCTSPlayer


def main(num_games=1000, max_iters=30, c_puct=1.0):
    """Play trivial-estimator MCTS players against each other on 3x6 NAC.

    Args:
        num_games: Games per evaluation call. Kept overridable so a smoke test
            can pass ``num_games=1``.
        max_iters: MCTS simulations per move.
        c_puct: PUCT exploration constant.
    """
    nac = games.NoughtsAndCrosses(3, 6)

    estimator = create_trivial_estimator(nac)

    players = {
        1: MCTSPlayer(nac, estimator, max_iters, c_puct),
        2: MCTSPlayer(nac, estimator, max_iters, c_puct),
    }

    players_switched = {1: players[2], 2: players[1]}

    player1_results_a, _ = evaluate(nac, players, num_games)
    player1_results_b, _ = evaluate(nac, players_switched, num_games)
    print(player1_results_a, player1_results_b)


if __name__ == "__main__":
    main()
