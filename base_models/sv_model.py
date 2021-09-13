import tensorflow as tf
# import numpy as np

def calculate_EER(cos_similarity, ground_truth):
    #@TODO implement the EER numpy function
    pass

class SpeakerVerificationModel(tf.keras.Model):
    def test_step(self, data):
        # Get two utterances and the label from the input
        x_1, x_2, y = data
        # Compute utterance embeddings from the input
        y_1_pred = self(x_1, training=False)
        y_2_pred = self(x_2, training=False)
        # Compute the cosine distances between the embeddings
        y_1_normal = tf.nn.l2_normalize(y_1_pred, 0)
        y_2_normal = tf.nn.l2_normalize(y_2_pred, 0)
        cos_similarity = tf.reduce_sum(tf.multiply(y_1_normal, y_2_normal))
        # Update the metrics.
        EER = tf.numpy_function(
                calculate_EER, [cos_similarity, y], tf.float, name=None
              )

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {"EER": EER}
