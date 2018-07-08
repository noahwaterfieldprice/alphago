from alphago.games import NoughtsAndCrosses
from alphago.estimator import NACNetEstimator
from alphago.alphago import train_alphago

from alphago.utilities import memoize_instance

learning_rate = 1e-4
game = NoughtsAndCrosses()
memoize_instance(game)


def create_estimator():
    return NACNetEstimator(learning_rate=learning_rate, action_indices=game.action_indices, l2_weight=0.1)


self_play_iters = 20
training_iters = 1000
evaluate_every = 5
alphago_steps = 2000
mcts_iters = 30
c_puct = 1.0
replay_length = 10000
num_evaluate_games = 30
win_rate = 0.8

checkpoint_path = 'checkpoints/'
pre_step = None

losses = train_alphago(game, create_estimator, self_play_iters=self_play_iters,
                       training_iters=training_iters,
                       checkpoint_path=checkpoint_path,
                       alphago_steps=alphago_steps,
                       evaluate_every=evaluate_every, batch_size=32,
                       mcts_iters=mcts_iters, c_puct=c_puct,
                       replay_length=replay_length,
                       num_evaluate_games=num_evaluate_games, win_rate=win_rate,
                       restore_step=None, verbose=True, summary_path="experiments/")
