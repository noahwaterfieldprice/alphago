import numpy as np
import tensorflow as tf

from alphago.mcts_tree import compute_distribution


def trivial_evaluator(state, next_states_function, action_space, is_terminal,
                      utility, which_player):
    """Evaluates a game state for a game. It is trivial in the sense that it
    returns the uniform probability distribution over all actions in the game.

    Parameters
    ----------
    state: tuple
        A state in the game.
    next_states_function: func
        Returns a dictionary from actions available in the current state to the
        resulting game states.
    action_space: list
        A list of all actions in the game.
    is_terminal: func
        Takes a game state and returns whether or not it is terminal.
    utility: func
        Given a terminal game state, returns an Outcome
    which_player: func
        Given a state, return whether it is player 1 or player 2 to play.

    Returns
    -------
    probs: dict
        A dictionary from actions to probabilities. Some actions might not be
        legal in this game state, but the evaluator returns a probability for
        choosing each one.
    value: float
        The evaluator's estimate of the value of the state 'state'.
    """

    if is_terminal(state):
        value = utility(state)
        value = value[which_player(state)]
        probs = {}
        return probs, value

    next_states = next_states_function(state)

    return compute_distribution({a: 1.0 for a in next_states}), 0.0


class BasicNACNet:
    def __init__(self):
        self.tensors = self._initialise_net()

    def _initialise_net(self):
        # TODO: test reshape recreates game properly
        state_vector = tf.placeholder(tf.float32, shape=(9,))
        pi = tf.placeholder(tf.float32, shape=(1, 9))
        outcomes = tf.placeholder(tf.float32, shape=(1, 1))

        input_layer = tf.reshape(state_vector, [-1, 3, 3, 1])

        conv1 = tf.layers.conv2d(inputs=input_layer, filters=5,
                                 kernel_size=[2, 2], padding='same',
                                 activation=tf.nn.relu)

        conv2 = tf.layers.conv2d(inputs=conv1, filters=5, kernel_size=[2, 2],
                                 padding='same', activation=tf.nn.relu)

        conv2_flat = tf.contrib.layers.flatten(conv2)

        dense = tf.layers.dense(inputs=conv2_flat, units=21,
                                activation=tf.nn.relu)

        values = tf.layers.dense(inputs=dense, units=1,
                                 activation=tf.nn.tanh)

        prob_logits = tf.layers.dense(inputs=dense, units=9)

        loss = tf.losses.mean_squared_error(outcomes, values) - \
               tf.tensordot(tf.transpose(pi), prob_logits, axes=1)

        tensors = [state_vector, outcomes, pi, values, prob_logits, loss]
        keys = "state_vector outcomes pi values prob_logits loss".split()
        return dict(zip(keys, tensors))

    def evaluate(self, state):

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            probs = sess.run(self.tensors['prob_logits'],
                     feed_dict={self.tensors['state_vector']: state})
            values = sess.run(self.tensors['values'],
                     feed_dict={self.tensors['state_vector']: state})

        return probs, values