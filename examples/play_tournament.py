import matplotlib

from alphago.elo import elo
from alphago.estimator import create_rollout_estimator, create_trivial_estimator
from alphago.evaluator import run_gauntlet
from alphago.games import ConnectFour
from alphago.player import MCTSPlayer, RandomPlayer

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import tqdm

tqdm.tqdm.monitor_interval = 0


def main(num_rounds=5, mcts_iters=10, output_path="results.png"):
    """Run a one-vs-all gauntlet on Connect Four and save an Elo/results plot.

    Args:
        num_rounds: Rounds played in the gauntlet (each pairing plays twice
            per round). Kept small so a smoke test can pass ``num_rounds=1``.
        mcts_iters: MCTS simulations per move for every MCTS player.
        output_path: Destination for the results heatmap image.
    """
    game = ConnectFour()

    trivial_estimator = create_trivial_estimator(game)
    rollout_estimator_10 = create_rollout_estimator(game, 10)
    rollout_estimator_100 = create_rollout_estimator(game, 100)

    mcts_args = mcts_iters, 0.5, 0.01
    random_player = RandomPlayer(game)
    trivial_mcts_player = MCTSPlayer(game, trivial_estimator, *mcts_args)
    rollout_mcts_player_10 = MCTSPlayer(game, rollout_estimator_10, *mcts_args)
    rollout_mcts_player_100 = MCTSPlayer(game, rollout_estimator_100, *mcts_args)

    players = {
        2: random_player,
        3: trivial_mcts_player,
        4: rollout_mcts_player_10,
    }

    results_list = run_gauntlet(
        game, (1, rollout_mcts_player_100), players, num_rounds
    )
    print(results_list)

    # Size the matrix to cover every player number, including the gauntlet
    # player (index 1), whose results would otherwise overflow a matrix sized
    # only by the opponents in ``players``.
    num_players = len(players) + 1
    results = np.zeros(shape=(num_players, num_players))
    for result in results_list:
        i, j, n = result
        results[i - 1, j - 1] = n

    fig, ax = plt.subplots()
    gammas = elo(results_list)
    ratings = [
        f"{player_no:d}: {gamma:.2f}" for player_no, gamma in sorted(gammas.items())
    ]
    ax.text(0.4, -0.8, str(ratings))
    a = ax.matshow(results, cmap=plt.cm.coolwarm)
    plt.colorbar(a)
    plt.tight_layout()
    fig.savefig(output_path)


if __name__ == "__main__":
    main()
