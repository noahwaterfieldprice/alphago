import time

from alphago.games import NoughtsAndCrosses, ConnectFour
from alphago.estimator import NACNetEstimator, ConnectFourNet
from alphago.alphago import train_alphago

from alphago.utilities import memoize_instance

learning_rate = 1e-3
game = ConnectFour()
game_name = 'connect_four'

#memoize_instance(game)


def create_estimator():
    return ConnectFourNet(learning_rate=learning_rate, action_indices=game.action_indices)

self_play_iters = 20
training_iters = 20000
evaluate_every = 10
alphago_steps = 1000
mcts_iters = 500
c_puct = 1.0
replay_length = 20000
num_evaluate_games = 50
win_rate = 0.6
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
