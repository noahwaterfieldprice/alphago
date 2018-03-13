import alphago.games.noughts_and_crosses as nac
from alphago import mcts_tree
from alphago.evaluator import create_trivial_evaluator, BasicNACNet
from alphago import comparator

if __name__ == "__main__":

    net1 = BasicNACNet()
    evaluator1 = net1.create_evaluator(nac.ACTION_INDICES)

    net2 = BasicNACNet()
    evaluator2 = net2.create_evaluator(nac.ACTION_INDICES)

    for i in range(10):
        # Train net 1, then compare against net 2.
        print("Training net 1")
        mcts_tree.alpha_go(evaluator1, net1.train, nac.ACTION_INDICES,
                           nac.compute_next_states, nac.INITIAL_STATE, nac.utility,
                           nac.which_player, nac.is_terminal, self_play_iters=100,
                           mcts_iters=100, c_puct=1.0)

        print("Comparing evaluators")
        evaluator1_wins, evaluator2_wins, draws = comparator.compare(
            nac.compute_next_states, nac.INITIAL_STATE, nac.utility,
            nac.which_player, nac.is_terminal, evaluator1, evaluator2,
            num_games=50)
        print("Evaluator 1 wins: {}, Evaluator 2 wins: {}, draws: "
              "{}".format(evaluator1_wins, evaluator2_wins, draws))
