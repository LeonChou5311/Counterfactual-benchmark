from tabnanny import verbose
from utils.preprocessing import DfInfo
from time import time

import numpy as np
import tensorflow as tf
import dice_ml
import pandas as pd
class Recorder:
    pass 

class RecordWrapper():
    '''
    Wrapper for decision tree and random forest.
    '''

    def __init__(self, model, cat_to_ohe_cat, ohe_feature_names):
        self.all_inputs = []
        self.model = model
        self.cat_to_ohe_cat = cat_to_ohe_cat
        self.ohe_feature_names = ohe_feature_names

    def dice_to_input(self, input_df):
        x = input_df.copy(deep=True)

        for k in self.cat_to_ohe_cat.keys():
            for ohe_col in self.cat_to_ohe_cat[k]:
                x[ohe_col] = x[k].apply(lambda v: 1 if v in ohe_col else 0)
            x.drop([k], axis=1, inplace=True)

        return np.array(x[self.ohe_feature_names])

    def predict_proba(self, x):
        # print("predict proba used in Record wrapper")
        self.all_inputs.append(x)
        cf_input = self.dice_to_input(x)
        return self.model.predict_proba(cf_input)

    def predict(self, x):
        # print("predict  used in Record wrapper")
        self.all_inputs.append(x)
        cf_input = self.dice_to_input(x)
        return self.model.predict(cf_input)


class NNRecordWrapper():
    '''
    Wrapper for NN specific.
    '''

    def __init__(self, model, cat_to_ohe_cat, ohe_feature_names):
        self.all_inputs = []
        self.model = model
        self.cat_to_ohe_cat = cat_to_ohe_cat
        self.ohe_feature_names = ohe_feature_names

    def dice_to_input(self, input_df):
        x = input_df.copy(deep=True)

        for k in self.cat_to_ohe_cat.keys():
            for ohe_col in self.cat_to_ohe_cat[k]:
                x[ohe_col] = x[k].apply(lambda v: 1 if v in ohe_col else 0)
            x.drop([k], axis=1, inplace=True)

        return np.array(x[self.ohe_feature_names])

    def predict(self, x):
        # print("predict being used in NN")
        self.all_inputs.append(x)
        cf_input = self.dice_to_input(x)
        return (self.model(tf.constant(cf_input.astype(float)), training=False) > 0.5).numpy().astype(int)[0]

    def predict_proba(self, x):
        # print("predict prob being used in NN")
        self.all_inputs.append(x)
        cf_input = self.dice_to_input(x)
        model_output = self.model(tf.constant(
            cf_input.astype(float)), training=False).numpy()
        # model_output = (model_output > 0.5).astype(float)
        return np.concatenate((1 - model_output, model_output), axis=1)


def dice_wrap_models(models, cat_to_ohe_cat, ohe_feature_names):
    '''
    Wrap the models to precess the input and output as the rquired of dice.
    '''
    # https://interpret.ml/DiCE/
    return {
        'dt': RecordWrapper(models['dt'], cat_to_ohe_cat, ohe_feature_names),
        'rfc': RecordWrapper(models['rfc'], cat_to_ohe_cat, ohe_feature_names),
        'nn': NNRecordWrapper(models['nn'], cat_to_ohe_cat, ohe_feature_names),
    }


def get_dice_cfs(data_interface, wrapped_models):
    '''
    Get DiCE instance for every wrapped models (wrapped by `dice_wrap_models` function).
    '''

    return {
        'dt': dice_ml.Dice(data_interface, dice_ml.Model(model=wrapped_models['dt'], backend="sklearn")),
        'rfc': dice_ml.Dice(data_interface, dice_ml.Model(model=wrapped_models['rfc'], backend="sklearn")),
        'nn': dice_ml.Dice(data_interface, dice_ml.Model(model=wrapped_models['nn'], backend="sklearn"))
    }


def generate_dice_result(df_info: DfInfo, test_df, models, num_instances, num_cf_per_instance, sample_size=200, test_start_idx=0, models_to_run=['dt', 'rfc', 'nn']):
    '''
    Generate counterfactuals using CounterfactualProto. 
    This counterfactul generating algorithms supports categorical features and numerical columns.

    [`df_info`] -> DfInfo instance containing all the data information required for generating counterfactuals.

    [`test_df`] -> Data frame contaning test data. (One-hot encoded format)

    [`models`] -> Dictionay of models (Usually containe <1> dt (Decision Tree) (2) rfc (RandomForest) (3) nn (Neural Network))
    [`num_instances`] -> Number of instances to generate counterfactuals. The instance is extracted from the testset. For example, 
    if `num_instances = 20`, it means the first 20 instances in the testset will be used for generating the counterfactuals.

    [`num_cf_per_instance`] -> Number of counterfactuals for each instance to generate. If `num_cf_per_instance = 5`, this function will
    run five times for each instance to search its counterfactual. Therefore, if you have `num_instances = 20, num_cf_per_instance = 5`, 100 searchings
    will be conducted. (Note: not promise that 100 counterfactuals will be found.)

    [`sample_size`] -> I found this parameters can be used for controlling the length of searching time, but not much information is provided on DiCE documentation.
    It's a parameters passed to generate_counterfactuals function.
    (http://interpret.ml/DiCE/dice_ml.explainer_interfaces.html?highlight=generate_counterfactuals#dice_ml.explainer_interfaces.explainer_base.ExplainerBase.generate_counterfactuals)
    '''

    d = dice_ml.Data(dataframe=df_info.scaled_df,
                     continuous_features=df_info.numerical_cols, outcome_name=df_info.target_name)

    wrapped_models = dice_wrap_models(
        models, df_info.cat_to_ohe_cat, df_info.ohe_feature_names)
    dice_cfs = get_dice_cfs(d, wrapped_models)

    Recorder.wrapped_models = wrapped_models
    Recorder.dice_cfs = dice_cfs

    results = {}

    for k in models_to_run:
        results[k] = []
        print(f"Finding counterfactual for {k}")
        for idx, instance in enumerate(df_info.scaled_df.iloc[test_df[test_start_idx:test_start_idx+num_instances].index].iloc):
            print(f"instance {idx}")
            for num_cf in range(num_cf_per_instance):
                print(f"CF {num_cf}")
                start_t = time()
                input_query = pd.DataFrame([instance.to_dict()])
                ground_truth = input_query[df_info.target_name][0]
                # print("Before generating cf")
                # Algorithm freeze here.
                # 1. perform this function outside of this script
                # 2. [prepare]: dice_cfs[k] (k is the running model) (x)
                # 3. [prepare]: input_query (x)
                # 4. sample_size = 50 (x)
                # 5. first try: get the successful case.
                # 6. get the error case.

                # Recorder.k = k
                # Recorder.input_query = input_query
                # Recorder.idx = idx
                # Recorder.ground_truth = ground_truth
                # print(f"Ground Truth is {input_query[df_info.target_name][0]}")
                # if idx == 3:
                #     raise StopIteration

                exp = dice_cfs[k].generate_counterfactuals(
                    input_query[df_info.feature_names], total_CFs=2, sample_size=sample_size, desired_class="opposite", verbose=True, posthoc_sparsity_param=None)

                # print("After generating cf")
                # dice_exp = dice_cfs['nn'].generate_counxwterfactuals(scaled_df.iloc[1:2], total_CFs=1, desired_class="opposite")
                # dice_exp.cf_examples_list[0].final_cfs_df.iloc[0][:-1]

                # if k=='nn':
                #     prediction = df_info.target_label_encoder.inverse_transform((wrapped_models[k].predict(input_query)[0]> 0.5).astype(int))[0]
                # else:
                prediction = df_info.target_label_encoder.inverse_transform(
                    wrapped_models[k].predict(input_query))[0]

                end_t = time()
                running_time = end_t - start_t
                results[k].append({
                    "input": input_query,
                    "cf": exp.cf_examples_list[0].final_cfs_df,
                    "running_time": running_time,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                })
    return results


def process_results(df_info: DfInfo, results):
    '''
    Process the result dictionary to construct data frames for each (dt, rfc, nn).
    '''

    result_dfs = {}

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

            origin_input_df[df_info.numerical_cols] = df_info.scaler.inverse_transform(
                origin_input_df[df_info.numerical_cols])
            origin_input_df.columns = origin_columns

            final_df = final_df.join([scaled_input_df, origin_input_df])

            # If counterfactaul found, inverse the scaling process to get the original data for cf.
            if not results[k][i]['cf'] is None:
                scaled_cf_df = results[k][i]['cf'].copy(deep=True)
                scaled_cf_df.loc[0, df_info.target_name] = df_info.target_label_encoder.inverse_transform(
                    [scaled_cf_df.loc[0, df_info.target_name]])[0]
                origin_cf_columns = [
                    f"origin_cf_{col}" for col in scaled_cf_df.columns]
                origin_cf_df = scaled_cf_df.copy(deep=True)
                scaled_cf_df.columns = [
                    f"scaled_cf_{col}" for col in scaled_cf_df.columns]

                origin_cf_df[df_info.numerical_cols] = df_info.scaler.inverse_transform(
                    origin_cf_df[df_info.numerical_cols])
                origin_cf_df.columns = origin_cf_columns

                final_df = final_df.join([scaled_cf_df, origin_cf_df])

            # Record additional information.
            final_df['running_time'] = results[k][i]['running_time']
            final_df['Found'] = "Y" if not results[k][i]['cf'] is None else "N"
            final_df['ground_truth'] = results[k][i]['ground_truth']
            final_df['prediction'] = results[k][i]['prediction']

            all_data.append(final_df)

        result_dfs[k] = pd.concat(all_data)

    return result_dfs
