from alphago.evaluator import trivial_evaluator
import alphago.noughts_and_crosses as nac
from alphago import mcts_tree

if __name__ == "__main__":

    max_iters = 1000
    c_puct = 1.0

    action_space = [(i, j) for i in range(3) for j in range(3)]

    def evaluator(state):
        return trivial_evaluator(
            state, nac.next_states, action_space, nac.is_terminal,
            nac.utility, nac.which_player)

    game_states, action_probs = mcts_tree.self_play(
        nac.next_states, evaluator, nac.INITIAL_STATE, nac.is_terminal,
        max_iters, c_puct)

    for state in game_states:
        nac.display(state)
        print("\n")
