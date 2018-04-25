from alphago.games import connect_four as cf


def test_can_play_connect_four():
    game = cf
    assert game.INITIAL_STATE

    game_state = game.INITIAL_STATE

    next_states = cf.compute_next_states(game_state)
    while len(next_states) > 0:
        action = min(next_states)
        game_state = next_states[action]
        print(action)
        next_states = cf.compute_next_states(game_state)
