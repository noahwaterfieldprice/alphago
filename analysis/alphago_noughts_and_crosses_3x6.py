import time

from alphago.games import NoughtsAndCrosses
from alphago.estimator import NAC3x6NetEstimator
from alphago.alphago import train_alphago
from alphago.utilities import memoize_instance

learning_rate = 0.1
game = NoughtsAndCrosses(rows=3, columns=6)
memoize_instance(game)
game_name = 'noughts_and_crosses'


def create_estimator():
    return NAC3x6NetEstimator(learning_rate=learning_rate,
                              action_indices=game.action_indices,
                              l2_weight=0.00001)


self_play_iters = 20
training_iters = 1000
evaluate_every = 5
alphago_steps = 2000
mcts_iters = 30
c_puct = 1.0
replay_length = 10000
num_evaluate_games = 50
win_rate = 0.55
batch_size = 32

current_time_format = time.strftime('experiment-%Y-%m-%d_%H:%M:%S')
path = "experiments/{}-{}/".format(game_name, current_time_format)
checkpoint_path = path + 'checkpoints/'
summary_path = path + 'logs/'

restore_step = None

losses = train_alphago(game, create_estimator, self_play_iters=self_play_iters,
                       training_iters=training_iters,
                       checkpoint_path=checkpoint_path,
                       summary_path=summary_path,
                       alphago_steps=alphago_steps,
                       evaluate_every=evaluate_every, batch_size=batch_size,
                       mcts_iters=mcts_iters, c_puct=c_puct,
                       replay_length=replay_length,
                       num_evaluate_games=num_evaluate_games, win_rate=win_rate,
                       restore_step=restore_step, verbose=True)
