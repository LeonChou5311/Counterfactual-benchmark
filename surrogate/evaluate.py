import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from utils.print import print_block
from surrogate import rules
import json


def get_origin_from_df(df, idx):
    origin_columns = [col for col in list(df.columns) if 'orgin_' in col]
    return np.array([list(df.iloc[idx][origin_columns])])


def get_cf_from_df(df, idx):
    cf_columns = [col for col in list(df.columns) if 'cf_' in col]
    return np.array([list(df.iloc[idx][cf_columns])])


def get_evaluation(df, feature_values, feature_names, surrogate_wrapper, scaler, generator):

    result_df = pd.DataFrame({}, columns=['input_query', 'cf', 'input_query_nn_pred', 'cf_nn_pred', 'permutation_nn_pred_portion', 'input_query_tree_pred', 'cf_tree_pred',
                             'input_query_decisions', 'cf_decisions', 'input_query_leaf_id', 'cf_leaf_id', 'dot_data', 'sparsity', 'distance', 'depth_diff', 'has_same_input_pred', 'has_same_cf_pred'])

    for idx in range(len(df)):
        print_block(idx, "Index")
        origin = get_origin_from_df(df, idx)
        cf = get_cf_from_df(df, idx)

        sparsity = np.sum((origin != cf).astype(int))
        dist = np.linalg.norm(origin-cf)

        original_nn_pred = surrogate_wrapper.predict(
            scaler.transform(origin)).numpy()[0][0]
        cf_nn_pred = surrogate_wrapper.predict(
            scaler.transform(cf)).numpy()[0][0]

        permuatations = generator.generate_data(origin, feature_values)

        permuataion_array = np.array(permuatations)
        permutation_pred = surrogate_wrapper.predict(
            scaler.transform(permuataion_array))
        _, portion = np.unique(np.round(permutation_pred), return_counts=True)

        tree = DecisionTreeClassifier(random_state=0)
        tree.fit(permuataion_array, np.round(permutation_pred.numpy()))

        origin_decisions, origin_tree_pred, origin_leaf_id = rules.get_decision_process(
            origin, tree, feature_names)

        cf_decisions, cf_tree_pred, cf_leaf_id = rules.get_decision_process(
            cf, tree, feature_names)

        depth_diff = abs(len(origin_decisions) - len(cf_decisions))

        dot_data = export_graphviz(
            tree,
            out_file=None,
            feature_names=feature_names,
            class_names=['No', 'Yes'],
            node_ids=True
        )

        temp_series = pd.Series({
            'input_query': origin.tolist()[0],
            'cf': cf.tolist()[0],
            'input_query_nn_pred': original_nn_pred,
            'cf_nn_pred': cf_nn_pred,
            'permutation_nn_pred_portion': portion.tolist(),
            'input_query_tree_pred': origin_tree_pred,
            'cf_tree_pred': cf_tree_pred,
            'input_query_decisions': json.dumps(origin_decisions),
            'cf_decisions': json.dumps(cf_decisions),
            'input_query_leaf_id': origin_leaf_id,
            'cf_leaf_id': cf_leaf_id,
            'dot_data': dot_data,
            'sparsity': sparsity,
            'distance': dist,
            'depth_diff': depth_diff,
            'has_same_input_pred': (np.round(original_nn_pred) == origin_tree_pred)[0],
            'has_same_cf_pred': (np.round(cf_nn_pred) == cf_tree_pred)[0],
        })

        result_df = result_df.append(temp_series, ignore_index=True)

    return result_df
