import pickle

import tensorflow as tf
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import os

def train_three_models(X_train, y_train):
    '''
    Construct and train ['dt', 'rfc', 'nn']

    ---
    Return -> A dictionary container three trained models.
    '''
    nn = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(24,activation='relu'),
                tf.keras.layers.Dense(12,activation='relu'),
                tf.keras.layers.Dense(12,activation='relu'),
                tf.keras.layers.Dense(12,activation='relu'),
                tf.keras.layers.Dense(12,activation='relu'),
                tf.keras.layers.Dense(1),
                tf.keras.layers.Activation(tf.nn.sigmoid),
            ]
        )
    nn.compile(optimizer="Adam", loss='binary_crossentropy', metrics=['accuracy'])
    nn.fit(X_train, y_train, batch_size=64, epochs=20, shuffle=True)

    models = {
        "dt": DecisionTreeClassifier().fit(X_train,y_train),
        "rfc": RandomForestClassifier().fit(X_train,y_train),
        "nn": nn,
    }


    return models



def train_three_models_lp(X_train, y_train):
    '''
    Construct and train ['dt', 'rfc', 'nn']

    ---
    Return -> A dictionary container three trained models.
    '''
    nn = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(24,activation='relu'),
                tf.keras.layers.Dense(12,activation='relu'),
                tf.keras.layers.Dense(1),
                tf.keras.layers.Activation(tf.nn.sigmoid),
            ]
        )
    nn.compile(optimizer="Adam", loss='binary_crossentropy', metrics=['accuracy'])
    nn.fit(X_train, y_train, batch_size=64, epochs=5, shuffle=True)

    models = {
        "dt": DecisionTreeClassifier(min_sample_leaf=10,max_depths=10).fit(X_train,y_train),
        "rfc": RandomForestClassifier(n_estimators=20, min_sample_leaf=10,max_depths=10).fit(X_train,y_train),
        "nn": nn,
    }


    return models



def evaluation_test(models, X_test, y_test):
    '''
    Evaluation the trained models.
    '''
    dt_acc = (models['dt'].predict(X_test) == y_test).astype(int).sum() / X_test.shape[0]
    rfc_acc = (models['rfc'].predict(X_test) == y_test).astype(int).sum() / X_test.shape[0]
    nn_acc = ((models['nn'].predict(X_test) > 0.5).flatten().astype(int) == y_test).astype(int).sum() / X_test.shape[0]

    print(f"DT: [{dt_acc:.4f}] | RF [{rfc_acc:.4f}] | NN [{nn_acc:.4f}]")


def save_three_models(models, dataset_name, path='./saved_models'):
    '''
    Save three trained models to desired `path`.
    '''
    storing_folder= f'{path}/{dataset_name}'
    os.makedirs(storing_folder, exist_ok=True)

    pickle.dump(models['dt'], open(f'{storing_folder}/dt.p', 'wb'))
    pickle.dump(models['rfc'], open(f'{storing_folder}/rfc.p', 'wb'))
    models['nn'].save(f'{storing_folder}/nn.h5',overwrite=True)

def save_lp_three_models(models, dataset_name, path='./saved_models'):
    '''
    Save three trained models to desired `path`.
    '''
    storing_folder= f'{path}/{dataset_name}'
    os.makedirs(storing_folder, exist_ok=True)

    pickle.dump(models['dt'], open(f'{storing_folder}/dt_lp.p', 'wb'))
    pickle.dump(models['rfc'], open(f'{storing_folder}/rfc_lp.p', 'wb'))
    models['nn'].save(f'{storing_folder}/nn_lp.h5',overwrite=True)


def load_three_models(num_features, dataset_name, path='./saved_models'):
    '''
    Load pre-trained model from the `path`.  Will be saved in `./saved_models` by default
    '''

    storing_folder= f'{path}/{dataset_name}'

    ### Load
    models = {}
    models['dt'] = pickle.load(open(f'{storing_folder}/dt.p', 'rb'))
    models['rfc'] = pickle.load(open(f'{storing_folder}/rfc.p', 'rb'))
    models['nn'] = tf.keras.models.load_model(f'{storing_folder}/nn.h5')

    ## Initialise NN output shape as (None, 1) for tensorflow.v1
    models['nn'].predict(np.zeros((2, num_features)))

    return models

def load_lp_three_models(num_features, dataset_name, path='./saved_models'):
    '''
    Load pre-trained model from the `path`.  Will be saved in `./saved_models` by default
    '''

    storing_folder= f'{path}/{dataset_name}'

    ### Load
    models = {}
    models['dt'] = pickle.load(open(f'{storing_folder}/dt_lp.p', 'rb'))
    models['rfc'] = pickle.load(open(f'{storing_folder}/rfc_lp.p', 'rb'))
    models['nn'] = tf.keras.models.load_model(f'{storing_folder}/nn_lp.h5')

    ## Initialise NN output shape as (None, 1) for tensorflow.v1
    models['nn'].predict(np.zeros((2, num_features)))

    return models

