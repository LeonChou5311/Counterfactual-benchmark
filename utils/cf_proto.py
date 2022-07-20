import numpy as np
import pandas as pd

from time import time
from utils.preprocessing import DfInfo
from utils.preprocessing import inverse_dummy, inverse_scaling
from alibi_cf.wrappers import AlibiBinaryPredictWrapper, AlibiBinaryNNPredictWrapper
from alibi.explainers import CounterfactualProto


'''s
Acronym:

dt -> Decision Tree
rfc -> Random Forest Classifier
nn -> Nueral Network
ohe -> One-hot encoding format
'''


class Recorder:
    pass


def get_cat_vars_info(cat_feature_names, train_df):
    '''
    Get information of categorical columns (one-hot).
    '''

    # Extract information of categorical features for alibi counterfactual prototype.
    cat_vars_idx_info = []
    for cat_col in cat_feature_names:
        num_unique_v = len(
            [col for col in train_df.columns if col.startswith(f"{cat_col}_")])
        first_index = min([list(train_df.columns).index(col)
                          for col in train_df.columns if col.startswith(f"{cat_col}_")])

        cat_vars_idx_info.append({
            "col": cat_col,
            "num_unique_v": num_unique_v,
            "first_index": first_index
        })

    # Encode the information to required format. { first_idx: num_unqiue_values }
    cat_vars_ohe = {}
    for idx_info in cat_vars_idx_info:
        cat_vars_ohe[idx_info['first_index']] = idx_info['num_unique_v']

    return cat_vars_idx_info, cat_vars_ohe


def alibi_wrap_models(models, output_int):
    '''
    Wrap the model to meet the requirements to Alibi.
    '''

    return {
        'dt': AlibiBinaryPredictWrapper(models['dt'], output_int=output_int),
        'rfc': AlibiBinaryPredictWrapper(models['rfc'], output_int=output_int),
        'nn': AlibiBinaryNNPredictWrapper(models['nn'], output_int=output_int),
    }


def get_proto_cfs(wrapped_models, feature_range, cat_vars_ohe, X_train, max_iters):
    '''
    Get CF generator.
    More information on CounterfactualProto -> (`https://docs.seldon.io/projects/alibi/en/latest/api/alibi.explainers.html?highlight=CounterFactualProto#alibi.explainers.CounterFactualProto`)
    '''

    proto_cfs = {}

    # Checking if the data contain catrgorical columns.
    cat_arguments = {}
    if len(cat_vars_ohe) > 0:
        cat_arguments['cat_vars'] = cat_vars_ohe

        # We use ohe (one-hot encoding) for categorical columns.
        cat_arguments['ohe'] = True

    for k in wrapped_models.keys():
        proto_cfs[k] = CounterfactualProto(
            wrapped_models[k].predict_proba,
            X_train[0].reshape(1, -1).shape,
            feature_range=feature_range,
            max_iterations=max_iters,
            **cat_arguments,  # Categorical infromation is pased here.
        )

        proto_cfs[k].fit(X_train)  # Fit the dataset.

    return proto_cfs


def generate_cf_proto_result(
        df_info: DfInfo,
        train_df,
        models,
        num_instances,
        num_cf_per_instance,
        X_train,
        X_test,
        y_test,
        max_iters=1000,
        models_to_run=['dt', 'rfc', 'nn'],
        output_int=True
):
    '''
    Generate counterfactuals using CounterfactualProto. 
    This counterfactul generating algorithms supports categorical features and numerical columns.

    [`df_info`] -> DfInfo instance containing all the data information required for generating counterfactuals.

    [`train_df`] -> Data frame contaning training data. (One-hot encoded format)

    [`models`] -> Dictionay of models (Usually containe <1> dt (Decision Tree) (2) rfc (RandomForest) (3) nn (Neural Network))
    [`num_instances`] -> Number of instances to generate counterfactuals. The instance is extracted from the testset. For example, 
    if `num_instances = 20`, it means the first 20 instances in the testset will be used for generating the counterfactuals.

    [`num_cf_per_instance`] -> Number of counterfactuals for each instance to generate. If `num_cf_per_instance = 5`, this function will
    run five times for each instance to search its counterfactual. Therefore, if you have `num_instances = 20, num_cf_per_instance = 5`, 100 searchings
    will be conducted. (Note: not promise that 100 counterfactuals will be found.)

    [`X_train, X_test, y_test`] -> Training and test data.

    [`max_iters`] -> Max iterations to run for searching a single counterfactual. It's a parameters in CounterfactualProto class. (`https://docs.seldon.io/projects/alibi/en/latest/api/alibi.explainers.html?highlight=CounterFactualProto#alibi.explainers.CounterFactualProto`)
    '''

    # Get all categorical columns names that are not label column.
    cat_feature_names = [
        col for col in df_info.categorical_cols if col != df_info.target_name]

    # Get one-hot encoding informations (Albii algorithm need it recognise categorical columns, or it will be treated as a numerical columns.)
    _, cat_vars_ohe = get_cat_vars_info(cat_feature_names, train_df)

    # Get wrapped models to meet the input and output of Alibi algorithms.
    wrapped_models = alibi_wrap_models(models, output_int)

    Recorder.wrapped_models = wrapped_models

    # Since we use min-max scaler and one-hot encoding, we can contraint the range in [0, 1]
    feature_range = (np.ones((1, len(df_info.feature_names))),
                     np.zeros((1, len(df_info.feature_names))))

    # Get counterfactual generator instance.
    profo_cfs = get_proto_cfs(
        wrapped_models, feature_range, cat_vars_ohe, X_train, max_iters)

    # Initialise the result dictionary.(It will be the return value.)
    results = {}

    # Loop through every models (dt, rfc, nn)
    for k in models_to_run:

        # Intialise the result for the classifier (predicting model).
        results[k] = []

        print(f"Finding counterfactual for {k}")

        # Looping throguh first `num_instances` in the test set.
        for idx, instance in enumerate(X_test[0:num_instances]):
            print(f"instance {idx}")

            # Reshape the input instance to make it 2D array (Which predictitive model accpeted) from 1D array.
            example = instance.reshape(1, -1)

            # Conduct the searching multiple (num_cf_per_instance) times  for a single instance.
            for num_cf in range(num_cf_per_instance):
                print(f"CF {num_cf}")

                start_t = time()
                exp = profo_cfs[k].explain(example)
                end_t = time()

                # Calculate the running time.
                running_time = end_t - start_t

                # Get the prediction from original predictive model in a human-understandable format.
                if k == 'nn':
                    # nn return float [0, 1], so we need to define a threshold for it. (It's usually 0.5 for most of the classifier).
                    prediction = df_info.target_label_encoder.inverse_transform(
                        (models[k].predict(example)[0] > 0.5).astype(int))[0]
                else:
                    # dt and rfc return int {1, 0}, so we don't need to define a threshold to get the final prediction.
                    prediction = df_info.target_label_encoder.inverse_transform(
                        models[k].predict(example))[0]

                # Checking if cf is found for this iteration.
                if (not exp.cf is None) and (len(exp.cf) > 0):
                    print("Found CF")
                    # Change the found CF from ohe format to original format.
                    cf = inverse_dummy(pd.DataFrame(
                        exp.cf['X'], columns=df_info.ohe_feature_names), df_info.cat_to_ohe_cat)

                    # Change the predicted value to the label we understand.
                    cf.loc[0, df_info.target_name] = df_info.target_label_encoder.inverse_transform([
                                                                                                    exp.cf['class']])[0]
                else:
                    print("CF not found")
                    cf = None

                # Change the found input from ohe format to original format.
                input_df = inverse_dummy(pd.DataFrame(
                    example, columns=df_info.ohe_feature_names), df_info.cat_to_ohe_cat)
                input_df.loc[0, df_info.target_name] = prediction

                results[k].append({
                    "input": input_df,
                    "cf": cf,
                    "running_time": running_time,
                    "ground_truth": df_info.target_label_encoder.inverse_transform([y_test[idx]])[0],
                    "prediction": prediction,
                })

    return results


def process_result(results, df_info):
    '''
    Process the result dictionary to construct data frames for each (dt, rfc, nn).
    '''

    results_df = {}

    # Loop through ['dt', 'rfc', 'nn']
    for k in results.keys():

        all_data = []
        for i in range(len(results[k])):

            final_df = pd.DataFrame([{}])

            # Inverse the scaling process to get the original data for input.
            scaled_input_df = results[k][i]['input'].copy(deep=True)
            origin_columns = [
                f"origin_input_{col}" for col in scaled_input_df.columns]
            origin_input_df = scaled_input_df.copy(deep=True)
            scaled_input_df.columns = [
                f"scaled_input_{col}" for col in scaled_input_df.columns]

            # origin_input_df[df_info.numerical_cols] = df_info.scaler.inverse_transform(
            #     origin_input_df[df_info.numerical_cols])

            origin_input_df = inverse_scaling(origin_input_df, df_info)

            origin_input_df.columns = origin_columns

            final_df = final_df.join([scaled_input_df, origin_input_df])

            # If counterfactaul found, inverse the scaling process to get the original data for cf.
            if not results[k][i]['cf'] is None:
                scaled_cf_df = results[k][i]['cf'].copy(deep=True)
                # Comment this
                # scaled_cf_df.loc[0, target_name] = target_label_encoder.inverse_transform([scaled_cf_df.loc[0, target_name]])[0]
                origin_cf_columns = [
                    f"origin_cf_{col}" for col in scaled_cf_df.columns]
                origin_cf_df = scaled_cf_df.copy(deep=True)
                scaled_cf_df.columns = [
                    f"scaled_cf_{col}" for col in scaled_cf_df.columns]

                # origin_cf_df[df_info.numerical_cols] = df_info.scaler.inverse_transform(
                #     origin_cf_df[df_info.numerical_cols])

                origin_cf_df = inverse_scaling(origin_cf_df, df_info)

                origin_cf_df.columns = origin_cf_columns

                final_df = final_df.join([scaled_cf_df, origin_cf_df])

            # Record additional information.
            final_df['running_time'] = results[k][i]['running_time']
            final_df['Found'] = "Y" if not results[k][i]['cf'] is None else "N"
            final_df['ground_truth'] = results[k][i]['ground_truth']
            final_df['prediction'] = results[k][i]['prediction']

            all_data.append(final_df)

        results_df[k] = pd.concat(all_data)

    return results_df
