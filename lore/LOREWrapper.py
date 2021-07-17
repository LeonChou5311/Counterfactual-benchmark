import tensorflow as tf

class LOREWarpper(tf.keras.Model):
    def __init__(self, model: tf.keras.Model):
        self.model = model
    def call(self, input):
        out = self.model(tf.constant(input))
        return out[:, 1:2]