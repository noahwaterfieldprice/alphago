import numpy as np
from tqdm import tqdm

from alphago import mcts_tree


def compare(compute_next_states, initial_state, utility, which_player,
            is_terminal, evaluator1, evaluator2, num_games):
    """Compare two evaluators. Returns the number of evaluator1 wins and number
    of draws in the games, as well as the total number of games.
    """
    evaluator1_wins = 0
    draws = 0

    # evaluator_player[1] is the player in the game corresponding to
    # evaluator1.
    evaluator_player = {1: 1, 2: 2}

    with tqdm(total=num_games) as pbar:
        for game_ix in range(num_games):
            # Start the game at the initial state. Player 1 in the game goes first.
            # This can either be evaluator1 or evaluator2.
            game_state = initial_state

            while not is_terminal(game_state):
                player = which_player(game_state)

                # Set the evaluator
                evaluator = evaluator1 if evaluator_player[1] == player else \
                    evaluator2

                # Create a new MCTS tree.
                # TODO: Potentially we want to keep the tree.
                root = mcts_tree.MCTSNode(game_state, player=player)

                # Use MCTS to compute action probabilities
                action_probs = mcts_tree.mcts(
                    root, evaluator, compute_next_states, utility, which_player,
                    is_terminal, mcts_iters=100, c_puct=1.0)

                # Sample an action
                actions, probs = zip(*action_probs.items())
                action_ix = np.random.choice(len(actions), p=probs)
                action = actions[action_ix]

                # The action_probs already incorporate the legal actions. Move to
                # the next game state.
                child_states = compute_next_states(game_state)
                game_state = child_states[action]

            # The state was terminal, so update win and draw counts.
            u = utility(game_state)
            if u[1] == 0:
                draws += 1
            else:
                # If evaluator1 wins, then increment evaluator1_wins.
                if u[evaluator_player[1]] > 0:
                    evaluator1_wins += 1

            # Change evaluator player.
            evaluator_player = {1: 2, 2: 1} if evaluator_player[1] == 1 else \
                {1: 1, 2: 2}
        
            games_so_far = game_ix + 1
            evaluator2_wins = games_so_far - draws - evaluator1_wins
            pbar.update(1)
            pbar.set_description("Win1/Win2/Draw: {}/{}/{}".format(
                evaluator1_wins, evaluator2_wins, draws))

    evaluator2_wins = num_games - draws - evaluator1_wins

    return evaluator1_wins, evaluator2_wins, draws
