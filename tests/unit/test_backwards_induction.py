from alphago import backwards_induction
from alphago.games import NoughtsAndCrosses
from alphago.games.noughts_and_crosses import GameState


def test_backwards_induction_on_nac_player1_forced_win():
    """A position where player 1 already has an unstoppable win.

    Player 1 holds the bottom-left and bottom-centre squares and it is
    player 1's move, so the position is a forced win for player 1 no matter
    which optimal move is taken. The solver must report a player-1 win and
    return a value-optimal action (several equally-optimal moves exist here,
    so we assert on the game-theoretic value rather than a single action).
    """
    nac = NoughtsAndCrosses()

    state = GameState(0b011000000, 0b000001000, 1)

    utility, best_action = backwards_induction.backwards_induction(nac, state)

    assert utility == {1: 1, 2: -1}
    # The returned action must itself be optimal, i.e. lead to the same value.
    child_state = nac.legal_actions(state)[best_action]
    child_utility, _ = backwards_induction.backwards_induction(nac, child_state)
    assert child_utility == utility


def test_backwards_induction_on_nac_draw_positions():
    """Backwards induction on two related noughts and crosses positions.

    From a single corner cross with player 2 to move, the centre is the
    unique drawing reply (every other reply loses for player 2), so we can
    assert the exact best action. After player 2 takes the centre, player 1
    can no longer force a win: every move draws, so we assert on the drawing
    value rather than a single arbitrary action.
    """
    nac = NoughtsAndCrosses()
    state = GameState(0b001000000, 0b000000000, 2)

    best_actions = {}
    value, best_action = backwards_induction.solve_game(best_actions, nac, state)

    # Center is the unique move that saves the draw for player 2.
    assert value == {1: 0, 2: 0}
    assert best_actions[state] == (1, 1)

    # After player 2 plays the centre, the game is a draw with optimal play.
    child_state = GameState(0b001000000, 0b000010000, 1)
    child_value, _ = backwards_induction.backwards_induction(nac, child_state)
    assert child_value == {1: 0, 2: 0}
