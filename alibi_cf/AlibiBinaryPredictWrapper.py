import numpy as np

class AlibiBinaryPredictWrapper():
    def __init__(self, model):
        self.model = model
        self.inputs = []

    def predict(self, x):
        num_instances = x.shape[0]
        self.inputs.append(x)
        modle_output = self.model.predict(x).reshape(num_instances, 1)
        return np.concatenate((modle_output ,1 - modle_output), axis=1)