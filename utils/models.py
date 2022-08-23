import pickle

import tensorflow as tf
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt


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
        # min_sample_leaf is not exist as an argument. go to documentation to check it.
        # it's called min_samples_leaf now.
        # so we do the same thing for rfc.
        "dt": DecisionTreeClassifier(min_samples_leaf=10,max_depth=10).fit(X_train,y_train), 
        "rfc": RandomForestClassifier(n_estimators=20, min_samples_leaf=10,max_depth=10).fit(X_train,y_train),
        "nn": nn,
    }

    return models


def evaluation_test(models, X_test, y_test):
    '''
    Evaluation the trained models.
    '''

    dt_pred = models['dt'].predict(X_test)
    rfc_pred = models['rfc'].predict(X_test)
    nn_pred = (models['nn'].predict(X_test) > 0.5).flatten().astype(int)

    # dt_acc = (models['dt'].predict(X_test) == y_test).astype(int).sum() / X_test.shape[0]
    # rfc_acc = (models['rfc'].predict(X_test) == y_test).astype(int).sum() / X_test.shape[0]
    # nn_acc = ((models['nn'].predict(X_test) > 0.5).flatten().astype(int) == y_test).astype(int).sum() / X_test.shape[0]


    #### DT model 
    print_eval_states(y_test, dt_pred, name="Decision Tree")
    print_eval_states(y_test, rfc_pred, name="Random Forest")
    print_eval_states(y_test, nn_pred, name="Neural Network")


def print_eval_states(y_test, y_pred, name=None):

    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    recall_score, precision_score, accuracy_score, f1_score
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Model: [{name}] | Accuracy: [{accuracy:.4f}] | Precision: [{precision:.4f} | Recall: [{recall:.4f}] | F1: [{f1:.4f}]")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
            plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title(f'Confusion Matrix ({name})', fontsize=18)
    plt.show()

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

