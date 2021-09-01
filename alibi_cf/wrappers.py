import numpy as np


class AlibiBinaryPredictWrapper():
    def __init__(self, model, output_int=False, num_possible_outputs=2):
        self.model = model
        self.inputs = []
        self.num_possible_outputs = num_possible_outputs
        self.output_int = output_int

    def predict(self, x):
        num_instances = x.shape[0]
        self.inputs.append(x)
        return self.model.predict(x).reshape(num_instances, 1)

    def predict_proba(self, x):
        num_instances = x.shape[0]
        self.inputs.append(x)
        model_output = self.model.predict_proba(x).reshape(num_instances, self.num_possible_outputs)
        if self.output_int:
            model_output = (model_output > 0.5).astype(int)
        return model_output


class AlibiBinaryNNPredictWrapper():
    def __init__(self, model, output_int=False) -> None:
        self.model = model
        self.inputs = []
        self.output_int = output_int

    def predict(self, x):
        num_instances = x.shape[0]
        self.inputs.append(x)

        model_output = self.model.predict(x).reshape(num_instances, 1)
        if self.output_int:
            model_output = (model_output > 0.5).astype(int)

        return model_output
        # modle_output = (self.model.predict(x).reshape(num_instances, 1) > 0.5).astype(int)
        # return np.concatenate((modle_output ,1 - modle_output), axis=1)

    def predict_proba(self, x):
        num_instances = x.shape[0]
        self.inputs.append(x)
        model_output = self.model.predict(x).reshape(num_instances, 1)
        if self.output_int:
            model_output = (model_output > 0.5).astype(int)
        return np.concatenate((1 - model_output, model_output), axis=1)
