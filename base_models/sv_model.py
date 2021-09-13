import tensorflow as tf
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d


def calculate_EER(cos_similarity, ground_truth):
    fpr, tpr, thresholds = roc_curve(ground_truth, cos_similarity, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh


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
        EER = tf.py_function(
                calculate_EER, [cos_similarity, y], tf.float, name=None
              )

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {"EER": EER}
