import numpy as np

def get_decision_process(x, tree, feature_names):

    node_indicator = tree.decision_path(x)

    node_index = node_indicator.indices[node_indicator.indptr[0]:
                                    node_indicator.indptr[1]]

    prediction = tree.predict(x)

    leaf_id = tree.apply(x)[0]

    decisions = []

    for node_id in node_index:
        ## Skip lift_id
        if leaf_id == node_id:
            continue

        if x[0, tree.tree_.feature[node_id]] <= tree.tree_.threshold[node_id]:
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        feature_name = feature_names[tree.tree_.feature[node_id]]
        feature_idx = tree.tree_.feature[node_id]
        value = x[0, tree.tree_.feature[node_id]]
        threshold = tree.tree_.threshold[node_id]

        decisions.append(
            {
                'node_id': int(node_id),
                'feature_idx': int(feature_idx),
                'feature_name': feature_name,
                'value': value,
                'threshold_sign': threshold_sign,
                'threshold': threshold,
            }
        )

        print("decision node {node} : (x [{feature}] = {value}) "
            "{inequality} {threshold})".format(
                node=node_id,
                feature=feature_name,
                value=value,
                inequality=threshold_sign,
                threshold=threshold))
    return decisions, prediction, leaf_id


def print_decisions(decisions):
    for d in decisions:
        print("decision node {node} : (x [{feature}] = {value}) "
            "{inequality} {threshold})".format(
                node=d['node_id'],
                feature=d['feature_name'],
                value=d['value'],
                inequality=d['threshold_sign'],
                threshold=d['threshold']))
