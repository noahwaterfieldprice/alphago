import argparse

from alphago.games import noughts_and_crosses as nac
from alphago.games import connect_four as cf
from alphago.alphago import train_alphago
from alphago.estimator import ConnectFourNet, NACNetEstimator


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('game', help="Either noughts_and_crosses or "
                                     "connect_four.")
    parser.add_argument('checkpoint_path',
                        help="The path to save checkpoints in. Include a "
                             "trailing slash.")

    args = parser.parse_args()

    learning_rate = 1e-4

    # Choose which game to play.
    if args.game == "noughts_and_crosses":
        game = nac

        def create_estimator():
            return NACNetEstimator(learning_rate=learning_rate)
    elif args.game == "connect_four":
        game = cf

        def create_estimator():
            return ConnectFourNet(learning_rate=learning_rate)
    else:
        raise ValueError("Game not implemented.")

    action_indices = game.ACTION_INDICES
    self_play_iters = 1000
    training_iters = 10000
    evaluate_every = 2
    alphago_steps = 1000
    mcts_iters = 50
    c_puct = 1.0
    replay_length = 10000
    num_evaluate_games = 50
    win_rate = 0.55

    train_alphago(game, create_estimator, self_play_iters=self_play_iters,
                  training_iters=training_iters,
                  checkpoint_path=args.checkpoint_path,
                  alphago_steps=alphago_steps,
                  evaluate_every=evaluate_every, batch_size=32,
                  mcts_iters=mcts_iters, c_puct=c_puct,
                  replay_length=replay_length,
                  num_evaluate_games=num_evaluate_games, win_rate=win_rate)
