import tensorflow as tf


class SummaryScalars:

    def __init__(self, scalar_names):

        # Create the tensorflow summary scalars
        graph = tf.Graph()
        self.sess = tf.Session(graph=graph)

        with graph.as_default():
            self.placeholders = {name: tf.placeholder(tf.float32, name=name)
                                 for name in scalar_names}
            self.summaries = {name: tf.summary.scalar(name, placeholder) for
                              name, placeholder in self.placeholders.items()}

            self.merged_summary = tf.summary.merge([summary for summary in
                                               self.summaries.values()])
            self.sess.run(tf.global_variables_initializer())

    def run(self, scalar_values, global_step, writer):
        """Runs the summary scalars.

        Parameters
        ----------
        scalar_values: dict
            Keys are scalar names and values are floats.
        global_step: int
            The global step to record the summaries at.
        """
        summary = self.sess.run(self.merged_summary, feed_dict={
            self.placeholders[name]: scalar_values[name] for name in
            scalar_values})

        writer.add_summary(summary, global_step)
