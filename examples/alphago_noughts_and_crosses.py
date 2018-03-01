import alphago.games.noughts_and_crosses as nac
from alphago import mcts_tree
from alphago.evaluator import create_trivial_evaluator, BasicNACNet

if __name__ == "__main__":

    max_iters = 1000
    c_puct = 1.0

    action_space = [(i, j) for i in range(3) for j in range(3)]
    action_indices = {a: action_space.index(a) for a in action_space}

    net = BasicNACNet()
    evaluator = net.create_evaluator(action_indices)

    all_training_data = []
    for i in range(100):
        game_states, action_probs = mcts_tree.self_play(
            nac.compute_next_states, evaluator, nac.INITIAL_STATE,
            nac.utility, nac.which_player, nac.is_terminal, max_iters, c_puct)

        training_data = mcts_tree.build_training_data(
            game_states, action_probs, nac.which_player, nac.utility,
            action_indices)
        all_training_data.extend(training_data)

        # print("Training data: {}".format(all_training_data))

        loss = net.train(all_training_data)
        print("Loss for net: {}".format(loss))
