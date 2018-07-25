import abc
import random

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from .games import Game


def create_trivial_estimator(game: Game):
    """Create a trivial evaluator function given a next states function
    for a game.

    Parameters
    ----------
    game:


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
        next_states = game.legal_actions(state)
        uniform_prior_probs = {action: 1 / len(next_states)
                               for action in next_states}
        return uniform_prior_probs, 0

    return trivial_estimator


def create_rollout_estimator(game, num_rollouts):
    # TODO: test this and write docstring
    def rollout_estimator(state):
        next_states = game.legal_actions(state)
        uniform_prior_probs = {action: 1 / len(next_states)
                               for action in next_states}
        player_no = game.current_player(state)
        total_value = 0
        for _ in range(num_rollouts):
            while not game.is_terminal(state):
                next_states = game.legal_actions(state)
                state = random.choice(list(next_states.values()))

            total_value += game.utility(state)[player_no]
        mean_value = total_value / num_rollouts

        return uniform_prior_probs, mean_value

    return rollout_estimator


class AbstractNeuralNetEstimator(abc.ABC):
    game_state_shape = NotImplemented
    action_indices = NotImplemented

    def __init__(self, learning_rate=1e-2, l2_weight=1e-4, value_weight=1):
        self.learning_rate = learning_rate
        self.l2_weight = l2_weight
        self.value_weight = value_weight
        self._initialise_net()

    @abc.abstractmethod
    def _initialise_net(self):
        """Initialise the neural network and all associated tensors."""

    @abc.abstractmethod
    def _state_to_vector(self, state):
        """Map the state to a vector suitable for input to the
        neural network estimator."""

    def __call__(self, state):
        """Returns the result of the neural net applied to the state. This is
        'probs' and 'value'

        Parameters
        ----------
        state: ndarray
            The input state to the network. Should be a numpy array.

        Returns
        -------
        probs: dict
            The probabilities returned by the net as a dictionary. The keys
            are the actions and the .
        value: ndarray
            The value returned by the net.
        """
        # Reshape the state if necessary so that it's 1 x game_state_shape. We
        # should only be evaluating one state at a time in this function.
        state = self._state_to_vector(state)

        # Evaluate the network at the state
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

        # value is currently an np array, so extract the float.
        probs = probs.ravel()
        [value] = value.ravel()

        # probs is currently an np array. Put the value into a
        # dictionary with keys the actions and values the probs.
        probs_dict = {action: probs[index] for
                      action, index in self.action_indices.items()}

        return probs_dict, value


    def loss(self, data, batch_size):
        """Computes the loss of the network on the data.

        Parameters
        ----------
        data: list
            A list consisting of (state, probs, z) tuples, where player is the
            player in the state and z is the utility to player in the last state
            from the corresponding self-play game.
        batch_size: int

        Returns
        -------
        loss: float
            The loss of the network on the given data.
        loss_value: float
            The loss of the value part of the network.
        loss_probs: float
            The loss of the probability part of the network.
        """
        iters = int(len(data) / batch_size)
        losses = []
        loss_value_list = []
        loss_probs_list = []
        for i in range(iters):
            batch_indices = range(i, i + batch_size)
            batch_data = [data[i] for i in batch_indices]

            # Set up the states, probs, zs arrays.
            states = np.array([x[0] for x in batch_data])
            pis = np.array([x[1] for x in batch_data])
            zs = np.array([x[2] for x in batch_data])
            zs = zs[:, np.newaxis]

            loss, loss_value, loss_probs = self.sess.run(
                [self.tensors['loss'], self.tensors['loss_value'],
                 self.tensors['loss_probs']], feed_dict={
                    self.tensors['state_vector']: states,
                    self.tensors['pi']: pis,
                    self.tensors['outcomes']: zs,
                    self.tensors['is_training']: False
                })
            losses.append(loss)
            loss_value_list.append(loss_value)
            loss_probs_list.append(loss_probs)

        return np.mean(losses), np.mean(loss_value_list), np.mean(loss_probs_list)

    def train_step(self, batch, return_summary=False):
        """Trains the network on the batch.

        Parameters
        ----------
        batch: list
            A list consisting of (state, probs, z) tuples, where player
            is the player in the state and z is the utility to player in
            the last state from the corresponding self-play game.
        return_summary: bool
            Whether to return the TensforFlow summary tensor for use in
            Tensorboard.
        Returns
        -------
        summary:
            The summary tensor, run on the batch.
        """
        # Set up the states, probs, zs arrays.
        states = np.array([x[0] for x in batch])
        pis = np.array([x[1] for x in batch])
        zs = np.array([x[2] for x in batch])
        zs = zs[:, np.newaxis]

        summary, value, probs, loss, _ = self.sess.run(
            [self.tensors['summary'], self.tensors['value'],
             self.tensors['probs'], self.tensors['loss'],
             self.train_op],
            feed_dict={self.tensors['state_vector']: states,
                       self.tensors['pi']: pis,
                       self.tensors['outcomes']: zs,
                       self.tensors['is_training']: True})

        # Update the global step
        self.global_step += 1
        if return_summary:
            return summary

    def train(self, training_data, batch_size, training_iters,
              mode='reinforcement', writer=None, verbose=True):
        """Trains the net on the training data.

        Parameters
        ----------
        training_data: list
            A list consisting of (state, probs, z) tuples, where player
            is the player in the state and z is the utility to player in
            the last state from the corresponding self-play game.
        batch_size: int
        training_iters: int
            The number of training iterations to run, where a training
            iteration corresponds to updating the net on a single batch
            of training data. If this is set to -1 in supervised mode,
            it will run for a whole epoch, i.e. process the entire data
            set exactly once.
        mode: str, {'reinforcement', 'supervised'}
            The mode of training. If running in reinforcement mode then
            the batch data are sampled randomly from the training data
            at each training iteration. If running in supervised mode,
            then the data is randomly ordered and then each training
            iteration steps through the data in batches.
        writer: tf.summary.FileWriter
            A FileWriter object for writing TensorFlow summaries to.
        verbose: bool
            Print out progress if True, else don't print anything.
        """

        if mode not in ['reinforcement', 'supervised']:
            raise ValueError("`mode` must be 'reinforcement', 'supervised'.")

        if mode == 'reinforcement':
            if training_iters == -1:
                raise ValueError("`training_iters` must be > 1 for "
                                 "reinforcement mode.")
            self._train_reinforcement(training_data, batch_size, training_iters,
                                      writer, verbose)
        elif mode == 'supervised':
            self._train_supervised(training_data, batch_size, training_iters,
                                   writer, verbose)

    def _train_reinforcement(self, training_data, batch_size, training_iters,
                             writer, verbose):
        """Train the net in reinforcement learning mode.

        In this case, a random batch is sampled for the data every
        training iteration. This may mean that the same data points are
        trained on multiple times before the every data point is in the
        training data is considered.
        """
        disable_tqdm = False if verbose else True
        for _ in tqdm(range(training_iters), disable=disable_tqdm):
            batch_indices = np.random.choice(len(training_data), batch_size)
            batch = [training_data[ix] for ix in batch_indices]
            summary = self.train_step(batch, return_summary=True)
            if writer is not None:
                writer.add_summary(summary, self.global_step)

    def _train_supervised(self, training_data, batch_size, training_iters,
                          writer, verbose):
        """Train the net in supervised learning mode.

        In this case, the training data are randomly shuffled and then
        they are processed sequentially in batches. The number of
        batches trained on is equal to the number training iterations.
        """
        size = len(training_data)
        training_indices = [i for i in range(size)]
        random.shuffle(training_indices)

        # calculate training iterations for single epoch if required
        if training_iters == -1:
            training_iters = (len(training_data) + batch_size - 1) // batch_size

        # generate batch indices, the final batch may be smaller if
        # `batch_size` doesn't evenly divide into size of training data
        batch_indices_list = [
            training_indices[i * batch_size:min(size, (i + 1) * batch_size)]
            for i in range(training_iters)]

        disable_tqdm = False if verbose else True
        for batch_indices in tqdm(batch_indices_list, disable=disable_tqdm):
            batch = [training_data[ix] for ix in batch_indices]
            summary = self.train_step(batch, return_summary=True)
            if writer is not None:
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

        return self.__call__

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

    def __init__(self, learning_rate, l2_weight, action_indices, value_weight=1):
        super().__init__(learning_rate, l2_weight, value_weight)
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

            regularizer = tf.contrib.layers.l2_regularizer(
                scale=self.l2_weight)
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

            conv3 = tf.contrib.layers.conv3d(
                inputs=conv1, num_outputs=16, kernel_size=[2, 2],
                stride=1, padding='SAME', weights_regularizer=regularizer)
            if use_batch_norm:
                conv3 = tf.contrib.layers.batch_norm(
                    conv3, is_training=is_training)
            conv3 = tf.nn.relu(conv3)

            conv3_flat = tf.contrib.layers.flatten(conv3)

            dense1 = tf.contrib.layers.fully_connected(
                inputs=conv3_flat, num_outputs=32,
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

            loss = self.value_weight * loss_value + loss_probs

            # Set up the training op
            self.train_op = \
                tf.train.MomentumOptimizer(self.learning_rate,
                                           momentum=0.9).minimize(loss)

            # Create summary variables for tensorboard
            loss_summary = tf.summary.scalar('loss', loss)

            summary = tf.summary.merge([loss_summary])

            self.sess.run(tf.global_variables_initializer())

            # Create a saver.
            self.saver = tf.train.Saver(max_to_keep=20)

        # Initialise global step (the number of training steps taken).
        self.global_step = 0

        tensors = [state_vector, outcomes, pi, value, prob_logits, probs,
                   loss, loss_value, loss_probs, is_training, summary]
        names = ("state_vector outcomes pi value prob_logits probs loss "
                 "loss_value loss_probs is_training summary").split()
        self.tensors = {name: tensor for name, tensor in zip(names, tensors)}

    def _state_to_vector(self, state):
        state = np.array(state).reshape((-1, 9))
        return np.nan_to_num(state)


class ConnectFourNet(AbstractNeuralNetEstimator):
    game_state_shape = (1, 42)

    def __init__(self, learning_rate, l2_weight, action_indices, value_weight=1):
        super().__init__(learning_rate, l2_weight, value_weight)
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

            regularizer = tf.contrib.layers.l2_regularizer(
                scale=self.l2_weight)
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

            loss = self.value_weight * loss_value + loss_probs

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
            self.saver = tf.train.Saver(max_to_keep=20)

        # Initialise global step (the number of training steps taken).
        self.global_step = 0

        tensors = [state_vector, outcomes, pi, value, prob_logits, probs,
                   loss, loss_value, loss_probs, is_training, summary]
        names = "state_vector outcomes pi value prob_logits probs loss " \
                "loss_value loss_probs is_training summary".split()
        self.tensors = {name: tensor for name, tensor in zip(names, tensors)}

    def _state_to_vector(self, state):
        return np.array(state).reshape((-1, 42))
