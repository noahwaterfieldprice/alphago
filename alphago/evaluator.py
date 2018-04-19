from tqdm import tqdm


def evaluate(game, players, num_games):
    """Compare two evaluators. Returns the number of evaluator1 wins
    and number of draws in the games, as well as the total number of
    games.
    """

    win, loss, draw = 1, -1, 0
    player1_results = {win: 0, loss: 0, draw: 0}

    with tqdm(total=num_games) as pbar:
        for game_no in range(num_games):
            game_state_list, _ = play(game, players)

            utility = game.utility(game_state_list[-1])
            player1_result = utility[1]

            player1_results[player1_result] += 1

            pbar.update(1)
            pbar.set_description("Win1/Win2/Draw: {}/{}/{}".format(
                player1_results[win], player1_results[loss],
                player1_results[draw]))

    return player1_results


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
    action_probs_list: list
        A list of action probability dictionaries, as returned by MCTS
        each time the algorithm has to take an action. The ith action
        probabilities dictionary corresponds to the ith game_state, and
        action_probs_list has length one less than game_state_list,
        since we don't have to move in a terminal state.
    """
    game_state = game.INITIAL_STATE
    game_state_list = [game_state]
    action_probs_list = []

    while not game.is_terminal(game_state):
        # First run MCTS to compute action probabilities.
        player_no = game.which_player(game_state)
        action, action_probs = players[player_no].choose_action(
            game_state, return_probabilities=True)

        # Play the action
        next_states = game.compute_next_states(game_state)
        game_state = next_states[action]

        # Add the action probabilities and game state to the list.
        action_probs_list.append(action_probs)
        game_state_list.append(game_state)

    return game_state_list, action_probs_list
