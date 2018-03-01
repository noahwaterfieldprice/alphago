import alphago.games.noughts_and_crosses as nac
from alphago import mcts_tree
from alphago.evaluator import create_trivial_evaluator, BasicNACNet

if __name__ == "__main__":

    net = BasicNACNet()
    evaluator = net.create_evaluator(nac.ACTION_INDICES)
    train_function = net.train

    mcts_tree.alpha_go(evaluator, train_function, nac.ACTION_INDICES,
                       nac.compute_next_states, nac.INITIAL_STATE, nac.utility,
                       nac.which_player, nac.is_terminal, self_play_iters=1000,
                       mcts_iters=100, c_puct=1.0)
