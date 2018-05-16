import time

from alphago.games import NoughtsAndCrosses
from alphago.estimator import NACNetEstimator
from alphago.alphago import train_alphago

from alphago.utilities import memoize_instance

learning_rate = 1e-4
game = NoughtsAndCrosses()

memoize_instance(game)

def create_estimator():
    return NACNetEstimator(learning_rate=learning_rate, action_indices=game.action_indices)

self_play_iters = 10
training_iters = 1000
evaluate_every = 2
alphago_steps = 1000
mcts_iters = 100
c_puct = 1.0
replay_length = 100000
num_evaluate_games = 20
win_rate = 0.8

current_time_format = time.strftime('experiment-%Y-%m-%d_%H:%M:%S')
path = "experiments/{}/".format(current_time_format)
checkpoint_path = path + 'checkpoints/'
summary_path = path + 'logs/'

restore_step = None

losses = train_alphago(game, create_estimator, self_play_iters=self_play_iters,
                       training_iters=training_iters,
                       checkpoint_path=checkpoint_path,
                       summary_path=summary_path,
                       alphago_steps=alphago_steps,
                       evaluate_every=evaluate_every, batch_size=32,
                       mcts_iters=mcts_iters, c_puct=c_puct,
                       replay_length=replay_length,
                       num_evaluate_games=num_evaluate_games, win_rate=win_rate,
                       restore_step=restore_step, verbose=True)