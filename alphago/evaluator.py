import numpy as np
import tensorflow as tf

from alphago.mcts_tree import compute_distribution


def create_trivial_evaluator(next_states_function):
    """Create a trivial evaluator function given a next states function
    for a game.

    Parameters
    ----------
    next_states_function: func
        Returns a dictionary from actions available in the current state to the
        resulting game states.

    Returns
    -------
    trivial_evaluator: func
        A function that returns a uniform probability distribution over all
        possible actions and a value given an input state.
    """

    def trivial_evaluator(state):
        """Evaluates a game state for a game. It is trivial in the sense
        that it returns the uniform probability distribution over all
        actions in the game.

        Parameters
        ----------
        state: tuple
            A state in the game.

        Returns
        -------
        prior_probs: dict
            A dictionary from actions to probabilities. Some actions might not be
            legal in this game state, but the evaluator returns a probability for
            choosing each one.
        value: float
            The evaluator's estimate of the value of the state 'state'.
         """
        next_states = next_states_function(state)
        return {action: 1 / len(next_states) for action in next_states}, 0

    return trivial_evaluator


class BasicNACNet:
    def __init__(self, learning_rate=1e-3):
        self.learning_rate = learning_rate
        self.tensors = self._initialise_net()

    def _initialise_net(self):
        # TODO: test reshape recreates game properly

        # Initialise a graph, session and saver for the net. This is so we can
        # use separate functions to run functions on the tensorflow graph. Using
        # 'with sess:' means you start with a new net each time.
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        # Use the graph to create the tensors
        with self.graph.as_default():
            state_vector = tf.placeholder(tf.float32, shape=(None, 9,))
            pi = tf.placeholder(tf.float32, shape=(None, 9))
            outcomes = tf.placeholder(tf.float32, shape=(None, 1))

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
            probs = tf.nn.softmax(logits=prob_logits)
            log_probs = tf.log(probs)

            loss_value = tf.losses.mean_squared_error(outcomes, values)
            loss_probs = -tf.reduce_mean(tf.multiply(pi, log_probs))
            loss = loss_value + loss_probs

            # Set up the training op
            self.train_op = \
                tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

            # Initialise all variables
            self.sess.run(tf.global_variables_initializer())

            # Create a saver.
            self.saver = tf.train.Saver()

        # Initialise global step (the number of training steps taken).
        self.global_step = 0

        tensors = [state_vector, outcomes, pi, values, prob_logits, probs,
                   loss, loss_value, loss_probs]
        names = "state_vector outcomes pi values prob_logits probs loss " \
                "loss_value loss_probs".split()
        return {name: tensor for name, tensor in zip(names, tensors)}

    def create_evaluator(self, action_indices):
        """Returns an evaluator function corresponding to the neural network.

        Parameters
        ----------
        action_indices: dict
            Dictionary with keys the available actions and values the index of
            that action. Indices must be unique in 0, 1, .., #actions-1.

        Returns
        -------
        evaluate: func
            A function that evaluates states.
        """

        def evaluate(state):
            # Reshape the state if necessary so that it's 1 x 9. We should only be
            # evaluating one state at a time in this function.
            state = np.reshape(state, (1, 9))
            state = np.nan_to_num(state)

            # Evaluate the network at the state
            probs, values = self.evaluate(state)

            # probs is currently an np array. Put the values into a dictionary with
            # keys the actions and values the probs.
            probs_dict = {action: probs[action_indices[action]] for action in
                          action_indices}

            return probs_dict, values[0]
        return evaluate

    def evaluate(self, state):
        """Returns the result of the neural net applied to the state. This is
        'probs' and 'values'

        Returns
        -------
        probs: np array
            The probabilities returned by the net.
        values: np array
            The value returned by the net.
        """

        probs = self.sess.run(
            self.tensors['probs'],
            feed_dict={self.tensors['state_vector']: state})
        values = self.sess.run(
            self.tensors['values'],
            feed_dict={self.tensors['state_vector']: state})

        return np.ravel(probs), values

    def train(self, training_data):
        """Trains the net on the training data.

        Parameters
        ----------
        training_data: list
            A list consisting of (state, probs, z) tuples, where player is the
            player in the state and z is the utility to player in the last state
            from the corresponding self-play game.
        """
        # Set up the states, probs, zs arrays.
        states = np.array([x[0] for x in training_data])
        pis = np.array([x[1] for x in training_data])
        zs = np.array([x[2] for x in training_data])
        zs = zs.reshape((-1, 1))

        values, probs, loss, _ = self.sess.run(
            [self.tensors['values'], self.tensors['probs'],
                self.tensors['loss'], self.train_op], feed_dict={
                    self.tensors['state_vector']: states,
                    self.tensors['pi']: pis,
                    self.tensors['outcomes']: zs,
                    })

        # Update the global step
        self.global_step += 1

        return loss

    def save(self, save_file):
        """Saves the net to save_file.
        """
        self.saver.save(self.sess, save_file, global_step=self.global_step)

    def restore(self, save_file):
        """Restore the net from save_file.
        """
        self.saver.restore(self.sess, save_file)
