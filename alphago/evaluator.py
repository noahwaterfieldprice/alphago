from tqdm import tqdm

from .alphago import self_play


def evaluate(game, players, num_games):
    """Compare two evaluators. Returns the number of evaluator1 wins
    and number of draws in the games, as well as the total number of
    games.
    """

    win, loss, draw = 1, -1, 0
    player1_results = {win: 0, loss: 0, draw: 0}

    with tqdm(total=num_games) as pbar:
        for game_no in range(num_games):
            game_state_list, _ = self_play(game, players)

            utility = game.utility(game_state_list[-1])
            player1_result = utility[1]

            player1_results[player1_result] += 1

            pbar.update(1)
            pbar.set_description("Win1/Win2/Draw: {}/{}/{}".format(
                player1_results[win], player1_results[loss],
                player1_results[draw]))

    return player1_results
