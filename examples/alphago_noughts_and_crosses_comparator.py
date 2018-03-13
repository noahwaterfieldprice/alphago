import alphago.games.noughts_and_crosses as nac
from alphago import mcts_tree
from alphago.evaluator import create_trivial_evaluator, BasicNACNet
from alphago import comparator

if __name__ == "__main__":

    net1 = BasicNACNet()
    evaluator1 = net1.create_evaluator(nac.ACTION_INDICES)

    evaluator2 = create_trivial_evaluator(nac.compute_next_states)

    mcts_iters = 20

    for i in range(50):
        # Train net 1, then compare against net 2.
        print("Training net 1")
        mcts_tree.alpha_go(evaluator1, net1.train, nac.ACTION_INDICES,
                           nac.compute_next_states, nac.INITIAL_STATE, nac.utility,
                           nac.which_player, nac.is_terminal, self_play_iters=100,
                           mcts_iters=mcts_iters, c_puct=1.0)

        print("We are player 1.")
        comparator.compare(
            nac.compute_next_states, nac.INITIAL_STATE, nac.utility,
            nac.which_player, nac.is_terminal, evaluator1, evaluator2,
            mcts_iters=mcts_iters, num_games=50)
        print("We are player 2.")
        comparator.compare(
            nac.compute_next_states, nac.INITIAL_STATE, nac.utility,
            nac.which_player, nac.is_terminal, evaluator2, evaluator1,
            mcts_iters=mcts_iters, num_games=50)
