import alphago.games.noughts_and_crosses as nac
from alphago import mcts_tree
from alphago.evaluator import create_trivial_evaluator

if __name__ == "__main__":

    max_iters = 1000
    c_puct = 1.0

    action_space = [(i, j) for i in range(3) for j in range(3)]

    evaluator = create_trivial_evaluator(nac.compute_next_states)

    game_states, action_probs = mcts_tree.self_play(
        nac.compute_next_states, evaluator, nac.INITIAL_STATE, nac.utility,
        nac.which_player, nac.is_terminal, max_iters, c_puct)

    for state in game_states:
        nac.display(state)
        print("\n")
