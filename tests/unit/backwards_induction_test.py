from alphago import backwards_induction
from alphago.games import NoughtsAndCrosses


def test_backwards_induction_on_nac_o_plays_top_right():
    nac = NoughtsAndCrosses()

    state = (1, 1, 0, -1, 0, 0, 0, 0, 0)

    utility, best_action = backwards_induction.backwards_induction(nac, state)

    assert best_action == (0, 2)


def test_backwards_induction_on_nac():
    nac = NoughtsAndCrosses()
    state = (1, 0, 0, 0, 0, 0, 0, 0, 0)

    best_actions = {}
    backwards_induction.solve_game(best_actions, nac, state)

    assert best_actions[state] == (1, 1)

    state = (1, 0, 0, 0, -1, 0, 0, 0, 0)

    assert best_actions[state] == (0, 1)
