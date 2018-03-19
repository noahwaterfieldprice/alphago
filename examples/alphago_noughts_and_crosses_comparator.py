import numpy as np
from tqdm import tqdm

import alphago.games.noughts_and_crosses as nac
import alphago.games.connect_four as cf
from alphago import alphago
from alphago import mcts_tree
from alphago.evaluator import create_trivial_evaluator, BasicNACNet
from alphago.evaluator import BasicConnectFourNet
from alphago import comparator
from alphago.backwards_induction import solve_game


def compare_against_optimal(evaluator, game, num_games, best_actions):
    player1_wins = 0
    player2_wins = 0
    draws = 0

    evaluator_player = 1
    with tqdm(total=num_games) as pbar:
        for game_ix in range(num_games):
            # Start the game at the initial state. Player 1 in the game goes
            #  first. This can either be evaluator1 or evaluator2.
            game_state = game.INITIAL_STATE

            while not game.is_terminal(game_state):
                player = game.which_player(game_state)

                if player == evaluator_player:
                    # Create a new MCTS tree.
                    root = mcts_tree.MCTSNode(game_state, player=player)

                    # Use MCTS to compute action probabilities
                    action_probs = mcts_tree.mcts(
                        root, evaluator, game, mcts_iters=mcts_iters,
                        c_puct=1.0)

                    # Sample an action
                    actions, probs = zip(*action_probs.items())
                    action_ix = np.random.choice(len(actions), p=probs)
                    action = actions[action_ix]
                else:
                    action = best_actions[game_state]

                # The action_probs already incorporate the legal actions.
                # Move to the next game state.
                child_states = game.compute_next_states(game_state)
                game_state = child_states[action]

            # The state was terminal, so update win and draw counts.
            u = game.utility(game_state)
            if u[1] == 0:
                draws += 1
            elif u[evaluator_player] > 0:
                # If player1 wins, then increment player1_wins.
                player1_wins += 1
            else:
                player2_wins += 1
                
            pbar.update(1)
            pbar.set_description("Win1/Win2/Draw: {}/{}/{}".format(
                player1_wins, player2_wins, draws))

            # Switch evaluator player
            evaluator_player = 1 if evaluator_player == 2 else 2

    return player1_wins, player2_wins, draws


if __name__ == "__main__":
    use_nac = True
    if use_nac:
        game = nac
        net1 = BasicNACNet()
        best_actions = {}
        solve_game(best_actions, game, game.INITIAL_STATE)
    else:
        game = cf
        net1 = BasicConnectFourNet()

    evaluator1 = net1.create_evaluator(game.ACTION_INDICES)
    evaluator2 = create_trivial_evaluator(game.compute_next_states)

    mcts_iters = 500

    for i in range(50):
        if use_nac:
            print("Comparing against optimal")
            num_games = 200
            compare_against_optimal(evaluator1, game, num_games, best_actions)

        # Train net 1, then compare against net 2.
        print("Training net 1")
        alphago.alphago(evaluator1, net1.train, game.ACTION_INDICES, game,
                        self_play_iters=600, mcts_iters=mcts_iters,
                        c_puct=1.0)

        print("We are player 1.")
        comparator.compare(game, evaluator1, evaluator2,
                           mcts_iters=mcts_iters, num_games=50)
        print("We are player 2.")
        comparator.compare(game, evaluator2, evaluator1,
                           mcts_iters=mcts_iters, num_games=50)
