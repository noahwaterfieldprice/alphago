from alphago import backwards_induction
from alphago.games import NoughtsAndCrosses


def test_backwards_induction_on_nac_o_plays_top_right():
    nac = NoughtsAndCrosses()

    state = (0b011000000, 0b000001000, 1)

    utility, best_action = backwards_induction.backwards_induction(nac, state)

    assert best_action == (2, 2)


def test_backwards_induction_on_nac():
    nac = NoughtsAndCrosses()
    state = (0b001000000, 0b000000000, 2)

    best_actions = {}
    backwards_induction.solve_game(best_actions, nac, state)
    print(best_actions)
    assert best_actions[state] == (1, 1)

    state = (0b001000000, 0b000010000, 1)

    assert best_actions[state] == (0, 1)
