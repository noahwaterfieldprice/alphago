import numpy as np

from alphago import backwards_induction
from alphago.games import noughts_and_crosses as nac


def test_backwards_induction_on_nac_o_plays_top_right():

    state = nac.INITIAL_STATE

    state = (1, 1, np.nan,
             -1, np.nan, np.nan,
             np.nan, np.nan, np.nan)

    utility, best_action = backwards_induction.backwards_induction(nac, state)

    assert best_action == (0, 2)


def test_backwards_induction_on_nac():
    state = (1, np.nan, np.nan,
             np.nan, np.nan, np.nan,
             np.nan, np.nan, np.nan)

    best_actions = {}
    backwards_induction.solve_game(best_actions, nac, state)

    assert best_actions[state] == (1, 1)

    state = (1, np.nan, np.nan,
             np.nan, -1, np.nan,
             np.nan, np.nan, np.nan)

    assert best_actions[state] == (0, 1)
