from tqdm import tqdm

from collections import namedtuple

GameLog = namedtuple("GameLog", "result actions game_states".split())


def evaluate(game, players, num_games, verbose=True):
    """Compare two evaluators. Returns the number of evaluator1 wins
    and number of draws in the games, as well as the total number of
    games.
    """

    win, loss, draw = 1, -1, 0
    player1_results = {win: 0, loss: 0, draw: 0}
    game_logs = []

    disable_tqdm = False if verbose else True
    with tqdm(total=num_games, disable=disable_tqdm) as pbar:
        for game_no in range(num_games):
            for player in players.values():
                player.reset()

            actions, game_states = play(game, players)

            utility = game.utility(game_states[-1])
            player1_result = utility[1]

            player1_results[player1_result] += 1

            pbar.update(1)
            pbar.set_description("Win1/Win2/Draw: {}/{}/{}".format(
                player1_results[win], player1_results[loss],
                player1_results[draw]))

            game_logs.append(GameLog(player1_result, actions, game_states))

    return player1_results, game_logs


def play(game, players):
    """Plays a two player game.

    Parameters
    ----------
    game: Game
        An object representing the game to be played.
    players: dict of Player
        An dictionary with keys the player numbers and values the players.

    Returns
    -------
    game_state_list: list
        A list of game states encountered in the self-play game. Starts
        with the initial state and ends with a terminal state.
    """
    game_state = game.initial_state
    game_states = [game_state]
    actions = []

    while not game.is_terminal(game_state):
        # first run MCTS to compute action probabilities.
        player_no = game.which_player(game_state)
        action = players[player_no].choose_action(game_state)

        # play the action
        next_states = game.compute_next_states(game_state)
        game_state = next_states[action]

        # update players with the played action
        for player in players.values():
            player.update(action)

        actions.append(action)
        game_states.append(game_state)

    return actions, game_states
