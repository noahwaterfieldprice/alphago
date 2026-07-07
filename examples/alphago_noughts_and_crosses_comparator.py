"""Compare a trivially-estimated MCTS player against the optimal player on
noughts and crosses, after a short AlphaGo self-play training run.

This demo exercises the full self-play training loop with tiny iteration
counts, then pits a trivial-estimator MCTS player against the exact optimal
player. Run it directly to override any hyperparameter via dotlist args, e.g.::

    uv run python examples/alphago_noughts_and_crosses_comparator.py \\
        training.alphago_steps=4 mcts.mcts_iters=25
"""

import sys

from alphago.alphago import train_alphago
from alphago.config import Config, load_config, resolve_paths
from alphago.estimator import NACNetEstimator, create_trivial_estimator
from alphago.evaluator import evaluate
from alphago.games.noughts_and_crosses import NoughtsAndCrosses
from alphago.player import MCTSPlayer, OptimalPlayer

# Tiny demo overrides so the self-play loop completes quickly. CLI dotlist
# args are appended after these, so a user override always wins.
#
# self_play_iters=20 clears the hardcoded 100-row `continue` guard in
# train_alphago (a NAC self-play game yields ~7 rows, so ~20 games give ~140
# rows). Below the threshold the loop trains nothing and the demo
# only ever compared two untrained players; above it a real optimise/evaluate
# step runs so the comparison reflects a trained network.
DEMO_OVERRIDES = [
    "training.alphago_steps=2",
    "training.self_play_iters=20",
    "training.training_iters=10",
    "training.evaluate_every=1",
    "training.num_evaluate_games=2",
    "mcts.mcts_iters=10",
]


def compare_against_optimal(game, player, player_no, num_games):
    """Evaluate ``player`` (playing as ``player_no``) against the optimal player."""
    optimal_player_no = 2 if player_no == 1 else 1
    optimal_player = OptimalPlayer(game)

    players = {player_no: player, optimal_player_no: optimal_player}

    return evaluate(game, players, num_games)


def main(cfg) -> None:
    """Train a short AlphaGo run then compare an MCTS player against optimal."""
    game = NoughtsAndCrosses()

    def create_estimator():
        return NACNetEstimator(
            learning_rate=cfg.estimator.learning_rate,
            l2_weight=cfg.estimator.l2_weight,
            value_weight=cfg.estimator.value_weight,
            action_indices=game.action_indices,
            device=cfg.estimator.device,
        )

    resolve_paths(cfg, "noughts_and_crosses")
    train_alphago(game, create_estimator, cfg, logger=None)

    trivial_estimator = create_trivial_estimator(game)
    player_no = 1
    player = MCTSPlayer(game, trivial_estimator, cfg.mcts.mcts_iters, cfg.mcts.c_puct)
    results, _ = compare_against_optimal(
        game, player, player_no, cfg.training.num_evaluate_games
    )

    # Print the comparison so the demo produces a visible result.
    wins, losses, draws = results[1], results[-1], results[0]
    print(f"MCTS player vs optimal — wins: {wins}, losses: {losses}, draws: {draws}")


if __name__ == "__main__":
    main(load_config(Config, DEMO_OVERRIDES + sys.argv[1:]))
