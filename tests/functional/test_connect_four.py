from alphago.games.connect_four import ConnectFour


def test_can_play_connect_four():
    game = ConnectFour()
    assert game.initial_state

    game_state = game.initial_state

    next_states = game.compute_next_states(game_state)
    while len(next_states) > 0:
        action = min(next_states)
        game_state = next_states[action]
        next_states = game.compute_next_states(game_state)
