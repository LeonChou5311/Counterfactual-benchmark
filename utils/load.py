from enum import Enum
from io import UnsupportedOperation
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from tensorflow.keras.models import model_from_json
from utils.exceptions import UnsupportedDataset

class SelectableDataset(Enum):
    Diabetes = "Diabetes"
    BreastCancer = "BreastCancer"


def get_dataset_path(dataset: SelectableDataset):
    if dataset == SelectableDataset.Diabetes:
        return './datasets/diabetes.csv'
    elif dataset == SelectableDataset.BreastCancer:
        return './datasets/breast_cancer.csv'
    else:
        raise UnsupportedDataset()


def get_dataset_target_names(dataset: SelectableDataset):
    if dataset == SelectableDataset.Diabetes:
        return 'Outcome'
    elif dataset == SelectableDataset.BreastCancer:
        return 'diagnosis'
    else:
        return UnsupportedDataset()


def encode_data(data, class_var):

    feature_names = data.drop([class_var], axis=1).columns.tolist()

    X = data[feature_names].values
    y = data[class_var].values

    n_features = X.shape[1]
    n_classes = len(data[class_var].unique())

    # create numerical encoding for attribute species
    enc = OneHotEncoder()
    Y = enc.fit_transform(y[:, np.newaxis]).toarray()

    # Scale data to have mean 0 and variance 1
    # which is importance for convergence of the neural network
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, Y, enc, scaler


def load_dataset(dataset: SelectableDataset):
    dataset_path = get_dataset_path(dataset)
    data = pd.read_csv(dataset_path)
    tar_name = get_dataset_target_names(dataset)
    feature_names = data.drop([tar_name], axis=1).columns.to_list()
    sampled_data = data.sample(frac=1)
    sampled_data = sampled_data[sampled_data[tar_name] == 0]

    no_data = sampled_data.sample(frac=1)[0:268]
    yes_data = data[data[tar_name] == 1]

    balanced_data = [no_data, yes_data]
    balanced_data = pd.concat(balanced_data)

    X, Y, encoder, scaler = encode_data(balanced_data, tar_name)

    n_features = X.shape[1]
    n_classes = len(data[tar_name].unique())

    return data, balanced_data, X, Y, encoder, scaler, n_features, n_classes, feature_names, tar_name


def load_training_data(dataset: SelectableDataset):
    dataset_path = get_dataset_path(dataset)
    X_train = pd.read_csv(dataset_path.replace(
        ".csv", "") + "_Xtrain.csv", index_col=False).values
    X_test = pd.read_csv(dataset_path.replace(
        ".csv", "") + "_Xtest.csv", index_col=False).values
    X_validation = pd.read_csv(dataset_path.replace(
        ".csv", "") + "_Xvalidation.csv", index_col=False).values
    Y_train = pd.read_csv(dataset_path.replace(
        ".csv", "") + "_Ytrain.csv", index_col=False).values
    Y_test = pd.read_csv(dataset_path.replace(
        ".csv", "") + "_Ytest.csv", index_col=False).values
    Y_validation = pd.read_csv(dataset_path.replace(
        ".csv", "") + "_Yvalidation.csv", index_col=False).values

    return X_train, Y_train, X_test, Y_test, X_validation, Y_validation


def get_model_path(dataset: SelectableDataset):
    if dataset == SelectableDataset.Diabetes:
        return f'./models/diabetes/model/'
    elif dataset == SelectableDataset.BreastCancer:
        return f'./models/breast_cancer/model/'
    else:
        return UnsupportedDataset()


def load_trained_model_for_dataset(dataset: SelectableDataset):
    model_name = 'model_h5_N12'
    model_path = get_model_path(dataset,)
    return load_model(model_name, model_path)

def load_model( model_name, path ):
    json_file = open( path + model_name +  "_DUO.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    # load weights into new model
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(path + model_name +  "_DUO.h5")
    print("Loaded model from disk")
    
    return loaded_model