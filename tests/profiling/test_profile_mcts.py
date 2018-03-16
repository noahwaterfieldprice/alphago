import alphago.games.noughts_and_crosses as nac
from alphago.alphago import self_play_multiple
from alphago.evaluator import create_trivial_evaluator


def test_self_play_multiple_can_play_nac():
    max_iters = 100
    num_self_play = 10
    c_puct = 1.0

    evaluator = create_trivial_evaluator(nac.compute_next_states)

    training_data = self_play_multiple(nac, evaluator, nac.ACTION_INDICES,
                                       max_iters, c_puct, num_self_play)
    assert len(training_data) > 0
