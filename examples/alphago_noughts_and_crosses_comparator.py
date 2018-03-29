import alphago.games.noughts_and_crosses as nac
from alphago.estimator import create_trivial_estimator
from alphago.evaluator import evaluate
from alphago.player import MCTSPlayer, OptimalPlayer


def compare_against_optimal(game, player, num_games):

    optimal_player_no = 1 if player.player_no == 2 else 2
    optimal_player = OptimalPlayer(optimal_player_no, game)

    players = {player.player_no: player,
               optimal_player.player_no: optimal_player}

    return evaluate(game, players, num_games)


if __name__ == "__main__":

    num_games = 200
    trivial_estimator = create_trivial_estimator(nac.compute_next_states)
    player = MCTSPlayer(1, nac, trivial_estimator, 10, 1)
    compare_against_optimal(nac, player, num_games)