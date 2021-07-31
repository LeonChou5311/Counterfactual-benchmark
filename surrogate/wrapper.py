import tensorflow as tf

class SurrogateWrapper():
    def __init__(self, model: tf.keras.Model):
        super(SurrogateWrapper, self).__init__()
        self.model = model
        self.all_input = []

    def predict(self, x):
        self.all_input.append(x)
        out = self.model(x)[:, 1:2]
        return out
        # return (out > .5).numpy().flatten().astype(int)