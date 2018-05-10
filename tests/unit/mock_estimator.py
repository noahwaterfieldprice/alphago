import numpy as np
import tensorflow as tf


class MockNetEstimator:
    def __init__(self, learning_rate, action_indices={i + 1: i for i in range(18)}):
        self._initialise_net()

    def _initialise_net(self):
        """Initiliase a 4-layer dense neural network for arbitrary
        input and output dimensions."""
        state_vector = tf.placeholder(tf.float32, shape=(1,))

        input_layer = tf.reshape(state_vector, [-1, 1])

        dense1 = tf.layers.dense(inputs=input_layer, units=20,
                                 activation=tf.nn.relu)

        dense2 = tf.layers.dense(inputs=dense1, units=20,
                                 activation=tf.nn.relu)

        values = tf.layers.dense(inputs=dense2, units=1,
                                 activation=tf.nn.tanh)

        prob_logits = tf.layers.dense(inputs=dense2, units=18)
        probs = tf.nn.softmax(logits=prob_logits)

        tensors = [state_vector, values, prob_logits, probs]
        names = "state_vector values prob_logits probs".split()
        self.tensors = {name: tensor for name, tensor in zip(names, tensors)}

    def __call__(self, state):
        """Returns the result of the neural net applied to the state. This is
        'probs' and 'value'

        Returns
        -------
        probs: array_like
            The probabilities returned by the net.
        value: array_like
            The value returned by the net.
        """
        if not hasattr(state, '__len__'):
            state = state,

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            probs = sess.run(
                self.tensors['probs'],
                feed_dict={self.tensors['state_vector']: state})
            [value] = sess.run(
                self.tensors['values'],
                feed_dict={self.tensors['state_vector']: state})

        return np.ravel(probs), value
