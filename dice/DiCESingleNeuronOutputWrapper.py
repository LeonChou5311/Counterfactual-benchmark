import tensorflow as tf

class DiCESingleNeuronOutputWrapper(tf.keras.layers.Layer):
    def __init__(self, model):
        super(DiCESingleNeuronOutputWrapper, self).__init__()
        self.model = model

    def call(self, inputs):
        return self.model(inputs)[:, 1:2]