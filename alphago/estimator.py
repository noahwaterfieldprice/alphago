import abc

import numpy as np
import tensorflow as tf
from tqdm import tqdm


def create_trivial_estimator(next_states_function):
    """Create a trivial evaluator function given a next states function
    for a game.

    Parameters
    ----------
    next_states_function: func
        Returns a dictionary from actions available in the current state to the
        resulting game states.

    Returns
    -------
    trivial_estimator: func
        A function that returns a uniform probability distribution over all
        possible actions and a value given an input state.
    """

    def trivial_estimator(state):
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

    return trivial_estimator


class AbstractNeuralNetEstimator(abc.ABC):

    def __init__(self, learning_rate=1e-2):
        self.learning_rate = learning_rate
        self._initialise_net()

    @abc.abstractmethod
    def _initialise_net(self):
        pass

    def __call__(self, state):
        """Returns the result of the neural net applied to the state. This is
        'probs' and 'value'

        Returns
        -------
        probs: ndarray
            The probabilities returned by the net.
        value: ndarray
            The value returned by the net.
        """

        probs = self.sess.run(
            self.tensors['probs'],
            feed_dict={
                self.tensors['state_vector']: state,
                self.tensors['is_training']: False})
        value = self.sess.run(
            self.tensors['value'],
            feed_dict={
                self.tensors['state_vector']: state,
                self.tensors['is_training']: False})

        return np.ravel(probs), value

    def train(self, training_data, batch_size, training_iters,
              writer, verbose=True):
        """Trains the net on the training data.

        Parameters
        ----------
        training_data: list
            A list consisting of (state, probs, z) tuples, where player is the
            player in the state and z is the utility to player in the last state
            from the corresponding self-play game.
        batch_size: int
        training_iters: int
        verbose: bool
            Print out progress if True, else don't print anything.
        """
        disable_tqdm = False if verbose else True
        for _ in tqdm(range(training_iters), disable=disable_tqdm):
            batch_indices = np.random.choice(len(training_data), batch_size,
                                             replace=True)
            batch_data = [training_data[i] for i in batch_indices]

            # Set up the states, probs, zs arrays.
            states = np.array([x[0] for x in batch_data])
            pis = np.array([x[1] for x in batch_data])
            zs = np.array([x[2] for x in batch_data])
            zs = zs[:, np.newaxis]

            summary, value, probs, loss, _ = self.sess.run(
                [self.tensors['summary'], self.tensors['value'],
                 self.tensors['probs'], self.tensors['loss'],
                 self.train_op], feed_dict={
                     self.tensors['state_vector']: states,
                     self.tensors['pi']: pis,
                     self.tensors['outcomes']: zs,
                     self.tensors['is_training']: True
                     })

            # Update the global step
            self.global_step += 1
        writer.add_summary(summary, self.global_step)

    def create_estimate_fn(self):
        """Returns an evaluator function corresponding to the neural network.

        Note that we expect self.action_indices to be a dictionary with keys
        the available actions and values the index of that action. Indices
        must be unique in 0, 1, .., #actions-1.

        Returns
        -------
        estimate: func
            A function that evaluates states.
        """

        def estimate_fn(state):
            # Reshape the state if necessary so that it's 1 x 9. We should
            # only be evaluating one state at a time in this function.
            state = np.reshape(state, self.game_state_shape)
            state = np.nan_to_num(state)

            # Evaluate the network at the state
            probs, [value] = self(state)

            # probs is currently an np array. Put the value into a
            # dictionary with keys the actions and values the probs.
            probs_dict = {action: probs[self.action_indices[action]] for
                          action in self.action_indices}

            return probs_dict, value

        return estimate_fn

    def save(self, save_file):
        """Saves the net to save_file.
        """
        self.saver.save(self.sess, save_file)

    def restore(self, save_file):
        """Restore the net from save_file.
        """
        self.saver.restore(self.sess, save_file)


class NACNetEstimator(AbstractNeuralNetEstimator):

    game_state_shape = (1, 9)

    def __init__(self, learning_rate, action_indices):
        super().__init__(learning_rate)
        self.action_indices = action_indices

    def _initialise_net(self):
        # TODO: test reshape recreates game properly

        # Initialise a graph, session and saver for the net. This is so we can
        # use separate functions to run functions on the tensorflow graph.
        # Using 'with sess:' means you start with a new net each time.
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        # Use the graph to create the tensors
        with self.graph.as_default():
            state_vector = tf.placeholder(tf.float32, shape=(None, 9,))
            pi = tf.placeholder(tf.float32, shape=(None, 9))
            outcomes = tf.placeholder(tf.float32, shape=(None, 1))

            input_layer = tf.reshape(state_vector, [-1, 3, 3, 1])

            l2_weight = 1e-4
            regularizer = tf.contrib.layers.l2_regularizer(scale=l2_weight)
            is_training = tf.placeholder(tf.bool)
            use_batch_norm = False

            conv1 = tf.contrib.layers.conv2d(
                inputs=input_layer, num_outputs=8, kernel_size=[2, 2],
                stride=1, padding='SAME', weights_regularizer=regularizer)
            if use_batch_norm:
                conv1 = tf.contrib.layers.batch_norm(
                    conv1, is_training=is_training)
            conv1 = tf.nn.relu(conv1)

            conv2 = tf.contrib.layers.conv2d(
                inputs=conv1, num_outputs=16, kernel_size=[2, 2],
                stride=1, padding='SAME', weights_regularizer=regularizer)
            if use_batch_norm:
                conv2 = tf.contrib.layers.batch_norm(
                    conv2, is_training=is_training)
            conv2 = tf.nn.relu(conv2)

            conv2_flat = tf.contrib.layers.flatten(conv2)

            dense1 = tf.contrib.layers.fully_connected(
                inputs=conv2_flat, num_outputs=32,
                weights_regularizer=regularizer)
            if use_batch_norm:
                dense1 = tf.contrib.layers.batch_norm(
                    dense1, is_training=is_training)
            dense1 = tf.nn.relu(dense1)

            value = tf.contrib.layers.fully_connected(
                inputs=dense1, num_outputs=1, weights_regularizer=regularizer,
                activation_fn=tf.nn.tanh)

            prob_logits = tf.contrib.layers.fully_connected(
                inputs=dense1, num_outputs=9, weights_regularizer=regularizer,
                activation_fn=None)
            probs = tf.nn.softmax(logits=prob_logits)

            # We want to compute log_probs = log(softmax(prob_logits)). This
            # simplifies to log_probs = prob_logits -
            # log(sum(exp(prob_logits))).
            log_sum_exp = tf.log(tf.reduce_sum(tf.exp(prob_logits), axis=1))
            log_probs = prob_logits - tf.expand_dims(log_sum_exp, 1)

            loss_value = tf.losses.mean_squared_error(outcomes, value)
            loss_probs = -tf.reduce_mean(tf.multiply(pi, log_probs))

            loss = loss_value + loss_probs

            # Set up the training op
            self.train_op = \
                tf.train.MomentumOptimizer(self.learning_rate,
                                           momentum=0.9).minimize(loss)

            # Create summary variables for tensorboard
            loss_summary = tf.summary.scalar('loss', loss)

            summary = tf.summary.merge([loss_summary])

            self.sess.run(tf.global_variables_initializer())

            # Create a saver.
            self.saver = tf.train.Saver()

        # Initialise global step (the number of training steps taken).
        self.global_step = 0

        tensors = [state_vector, outcomes, pi, value, prob_logits, probs,
                   loss, loss_value, loss_probs, is_training, summary]
        names = ("state_vector outcomes pi value prob_logits probs loss "
                 "loss_value loss_probs is_training summary").split()
        self.tensors = {name: tensor for name, tensor in zip(names, tensors)}


class ConnectFourNet(AbstractNeuralNetEstimator):
    game_state_shape = (1, 42)

    def __init__(self, learning_rate, action_indices):
        super().__init__(learning_rate)
        self.action_indices = action_indices

    def _initialise_net(self):
        # TODO: test reshape recreates game properly

        # Initialise a graph, session and saver for the net. This is so we can
        # use separate functions to run functions on the tensorflow graph.
        # Using 'with sess:' means you start with a new net each time.
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        # Use the graph to create the tensors
        with self.graph.as_default():
            state_vector = tf.placeholder(tf.float32, shape=(None, 42,))
            pi = tf.placeholder(tf.float32, shape=(None, 7))
            outcomes = tf.placeholder(tf.float32, shape=(None, 1))

            input_layer = tf.reshape(state_vector, [-1, 6, 7, 1])

            l2_weight = 1e-4
            regularizer = tf.contrib.layers.l2_regularizer(scale=l2_weight)
            is_training = tf.placeholder(tf.bool)

            conv1 = tf.contrib.layers.conv2d(
                inputs=input_layer, num_outputs=8, kernel_size=[3, 3],
                padding='SAME', weights_regularizer=regularizer,
                activation_fn=tf.nn.relu)

            conv2 = tf.contrib.layers.conv2d(
                inputs=conv1, num_outputs=16, kernel_size=[3, 3],
                padding='SAME', weights_regularizer=regularizer,
                activation_fn=tf.nn.relu)

            conv3 = tf.contrib.layers.conv2d(
                inputs=conv2, num_outputs=32, kernel_size=[3, 3],
                padding='SAME', weights_regularizer=regularizer,
                activation_fn=tf.nn.relu)

            conv4 = tf.contrib.layers.conv2d(
                inputs=conv3, num_outputs=64, kernel_size=[3, 3],
                padding='SAME', weights_regularizer=regularizer,
                activation_fn=tf.nn.relu)

            conv4_flat = tf.contrib.layers.flatten(conv4)

            dense1 = tf.contrib.layers.fully_connected(
                inputs=conv4_flat, num_outputs=64,
                weights_regularizer=regularizer, activation_fn=tf.nn.relu)

            dense2 = tf.contrib.layers.fully_connected(
                inputs=dense1, num_outputs=128, weights_regularizer=regularizer,
                activation_fn=tf.nn.relu)

            dense3 = tf.contrib.layers.fully_connected(
                inputs=dense2, num_outputs=256, weights_regularizer=regularizer,
                activation_fn=tf.nn.relu)

            value = tf.contrib.layers.fully_connected(
                inputs=dense3, num_outputs=1, weights_regularizer=regularizer,
                activation_fn=tf.nn.tanh)

            prob_logits = tf.contrib.layers.fully_connected(
                inputs=dense3, num_outputs=7, weights_regularizer=regularizer,
                activation_fn=None)
            probs = tf.nn.softmax(logits=prob_logits)

            # We want to compute log_probs = log(softmax(prob_logits)). This
            # simplifies to log_probs = prob_logits -
            # log(sum(exp(prob_logits))).
            log_sum_exp = tf.log(tf.reduce_sum(tf.exp(prob_logits), axis=1))
            log_probs = prob_logits - tf.expand_dims(log_sum_exp, 1)

            loss_value = tf.losses.mean_squared_error(outcomes, value)
            loss_probs = -tf.reduce_mean(tf.multiply(pi, log_probs))

            loss = loss_value + loss_probs

            # Set up the training op
            self.train_op = \
                tf.train.MomentumOptimizer(self.learning_rate,
                                           momentum=0.9).minimize(loss)

            # Create summary variables for tensorboard
            loss_summary = tf.summary.scalar('loss', loss)

            summary = tf.summary.merge([loss_summary])

            # Initialise all variables
            self.sess.run(tf.global_variables_initializer())

            # Create a saver.
            self.saver = tf.train.Saver()

        # Initialise global step (the number of training steps taken).
        self.global_step = 0

        tensors = [state_vector, outcomes, pi, value, prob_logits, probs,
                   loss, loss_value, loss_probs, is_training, summary]
        names = "state_vector outcomes pi value prob_logits probs loss " \
                "loss_value loss_probs is_training summary".split()
        self.tensors = {name: tensor for name, tensor in zip(names, tensors)}
