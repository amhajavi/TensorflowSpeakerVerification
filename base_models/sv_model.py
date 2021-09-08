import tensorflow as tf


class SpeakerVerificationModel(tf.keras.Model):
    def test_step(self, data):
        # Unpack the data
        # x_1, x_2 , y = data
        # Compute predictions
        # y_1_pred = self(x_1, training=False)
        # y_2_pred = self(x_2, training=False)
        # Updates the metrics tracking the loss
        # Update the metrics.
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {"anything I say": 0.1}
