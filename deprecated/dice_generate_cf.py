import tensorflow as tf
import pandas as pd
import numpy as np

from utils.df_loader import load_adult_df, load_compas_df, load_german_df, load_diabetes_df
from utils.preprocessing import preprocess_df
from sklearn.model_selection import train_test_split
from utils.dice import generate_dice_result, process_results
from utils.models import train_three_models, evaluation_test, save_three_models, load_three_models
from utils.save import save_result_as_csv

pd.options.mode.chained_assignment = None 

print('TF version: ', tf.__version__)
print('Eager execution enabled: ', tf.executing_eagerly()) # False

seed = 123
tf.random.set_seed(seed)
np.random.seed(seed)


#### Select dataset ####
dataset_name = 'diabetes' # [adult, german, compas]

if dataset_name == 'adult':
    dataset_loading_fn = load_adult_df
elif dataset_name == 'german':
    dataset_loading_fn = load_german_df
elif dataset_name == 'compas':
    dataset_loading_fn = load_compas_df
elif dataset_name == 'diabetes':
    dataset_loading_fn = load_diabetes_df
else:
    raise Exception("Unsupported dataset")


#### Load datafram info.
df_info = preprocess_df(dataset_loading_fn)


### Seperate to train and test set.
train_df, test_df = train_test_split(df_info.dummy_df, train_size=.8, random_state=seed, shuffle=True)

### Get training and testing array.
X_train = np.array(train_df[df_info.ohe_feature_names])
y_train = np.array(train_df[df_info.target_name])
X_test = np.array(test_df[df_info.ohe_feature_names])
y_test = np.array(test_df[df_info.target_name])

### Train models.
# models = train_three_models(X_train, y_train)

### Save models.
# save_three_models(models, dataset_name)

### Load models.
models = load_three_models(X_train.shape[-1], dataset_name)

### Print out accuracy on testset.
evaluation_test(models, X_test, y_test)

### Setting up the CF generating amount.
num_instances = 5
num_cf_per_instance = 1

### Generate CF
results = generate_dice_result(df_info, test_df, models, num_instances, num_cf_per_instance, sample_size=50)
result_dfs = process_results(df_info, results)

### Save result as file.
save_result_as_csv("dice", dataset_name, result_dfs)