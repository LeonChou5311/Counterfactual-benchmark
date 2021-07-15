from utils.prediction import PredictionTypeWrapper, PredictionTypeWrapper, generate_local_predictions, wrap_information
from dice import DiCECounterfactaulWrapper, DiCESingleNeuronOutputWrapper
from utils.cf import generate_cf_for_all
from utils import load

import tensorflow as tf
import dice_ml
import numpy as np
import pandas as pd

### Set random seed
seed = 123
tf.random.set_seed(seed)
np.random.seed(seed)

### Load data
dataset = load.SelectableDataset.Diabetes
data, balanced_data, X, Y, encoder, scaler, n_features, n_classes, feature_names, target_name = load.load_dataset(dataset)
X_train, Y_train, X_test, Y_test, X_validation, Y_validation = load.load_training_data(dataset)
model = load.load_trained_model_for_dataset(dataset)

### Seperate to different predictive type
diabetes_feature_range = (X_train.min(axis=0), X_train.max(axis=0))
local_data_dict = generate_local_predictions( X_test, Y_test, model, scaler, encoder )
true_positives,true_negatives, false_positives, false_negatives = wrap_information( local_data_dict )
all_predictions = PredictionTypeWrapper(true_positives,true_negatives, false_positives, false_negatives)

### Initialise DiCE
temp_df = pd.DataFrame(X, columns=feature_names)
temp_df[target_name] = Y[:, 1]
d = dice_ml.Data(dataframe=balanced_data, continuous_features=feature_names, outcome_name=target_name)
wrapped_model = DiCESingleNeuronOutputWrapper(model)
m = dice_ml.Model(model=wrapped_model, backend="TF2")
exp = dice_ml.Dice(d,m)

### Generate single counterfactual
dice_wrapper = DiCECounterfactaulWrapper(exp, feature_names)
dice_wrapper.run_counterfactual_print_result(all_predictions.get_true_negative(3))

### Generate counterfactuals for all
dice_cf_df = generate_cf_for_all(
    all_predictions,
    dice_wrapper.run_counterfactual_print_result,
    feature_names
    )