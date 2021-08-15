from utils.preprocessing import DfInfo
from time import time

import numpy as np
import tensorflow as tf
import dice_ml
import pandas as pd

class RecordWrapper():
    def __init__(self, model, all_cat_ohe_cols, ohe_feature_names):
        self.all_inputs = []
        self.model = model
        self.all_cat_ohe_cols = all_cat_ohe_cols
        self.ohe_feature_names = ohe_feature_names

    def dice_to_input(self, input_df):
        x = input_df.copy(deep=True)

        for k in self.all_cat_ohe_cols.keys():
            for ohe_col in self.all_cat_ohe_cols[k]:
                x[ohe_col] = x[k].apply(lambda v: 1 if v in ohe_col else 0)
            x.drop([k], axis=1, inplace=True)

        return np.array(x[self.ohe_feature_names])

    def predict_proba(self, x):
        self.all_inputs.append(x)
        cf_input = self.dice_to_input(x)
        return self.model.predict_proba(cf_input)

    def predict(self, x):
        self.all_inputs.append(x)
        cf_input = self.dice_to_input(x)
        return self.model.predict(cf_input)


class NNRecordWrapper():
    def __init__(self, model, all_cat_ohe_cols, ohe_feature_names):
        self.all_inputs = []
        self.model = model
        self.all_cat_ohe_cols = all_cat_ohe_cols
        self.ohe_feature_names = ohe_feature_names

    def dice_to_input(self, input_df):
        x = input_df.copy(deep=True)

        for k in self.all_cat_ohe_cols.keys():
            for ohe_col in self.all_cat_ohe_cols[k]:
                x[ohe_col] = x[k].apply(lambda v: 1 if v in ohe_col else 0)
            x.drop([k], axis=1, inplace=True)

        return np.array(x[self.ohe_feature_names])

    def predict(self, x):
        self.all_inputs.append(x)
        cf_input = self.dice_to_input(x)
        return self.model.predict(tf.constant(cf_input.astype(float)))

    def predict_proba(self, x):
        self.all_inputs.append(x)
        cf_input = self.dice_to_input(x)
        return self.model.predict(tf.constant(cf_input.astype(float)))


def dice_wrap_models(models, all_cat_ohe_cols, ohe_feature_names):
    return {
        'dt': RecordWrapper(models['dt'], all_cat_ohe_cols, ohe_feature_names),
        'rfc': RecordWrapper(models['rfc'], all_cat_ohe_cols, ohe_feature_names),
        'nn': NNRecordWrapper(models['nn'], all_cat_ohe_cols, ohe_feature_names),
    }

def get_dice_cfs(data_interface, wrapped_models):
    return {
        'dt': dice_ml.Dice(data_interface, dice_ml.Model(model=wrapped_models['dt'], backend="sklearn")),
        'rfc': dice_ml.Dice(data_interface, dice_ml.Model(model=wrapped_models['rfc'], backend="sklearn")),
        'nn': dice_ml.Dice(data_interface, dice_ml.Model(model=wrapped_models['nn'], backend="sklearn"))
    }


def generate_dice_result(df_info: DfInfo, test_df, models, num_instances, num_cf_per_instance, sample_size=200):

    d = dice_ml.Data(dataframe=df_info.scaled_df, continuous_features=df_info.numerical_cols, outcome_name=df_info.target_name)

    wrapped_models = dice_wrap_models(models, df_info.all_cat_ohe_cols, df_info.ohe_feature_names)
    dice_cfs = get_dice_cfs(d, wrapped_models)

    results = {}

    for k in dice_cfs.keys():
        results[k] = []
        print(f"Finding counterfactual for {k}")
        for idx, instance in enumerate(df_info.scaled_df.iloc[test_df[0:num_instances].index].iloc):
            print(f"instance {idx}")
            for num_cf in range(num_cf_per_instance):
                print(f"CF {num_cf}")
                start_t = time()

                input_query = pd.DataFrame([instance.to_dict()])
                ground_truth = input_query[df_info.target_name][0]
                exp = dice_cfs[k].generate_counterfactuals(input_query, total_CFs=1, sample_size=sample_size, desired_class="opposite")

                # dice_exp = dice_cfs['nn'].generate_counterfactuals(scaled_df.iloc[1:2], total_CFs=1, desired_class="opposite")
                # dice_exp.cf_examples_list[0].final_cfs_df.iloc[0][:-1]

                if k=='nn':
                    prediction = df_info.target_label_encoder.inverse_transform((wrapped_models[k].predict(input_query)[0]> 0.5).astype(int))[0]
                else:
                    prediction = df_info.target_label_encoder.inverse_transform(wrapped_models[k].predict(input_query))[0]
                
                end_t = time ()
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

    result_dfs = {}

    for k in results.keys():

        all_data = []

        for i in range(len(results[k])):
            final_df = pd.DataFrame([{}])

            scaled_input_df = results[k][i]['input'].copy(deep=True)
            origin_columns = [f"origin_input_{col}"  for col in scaled_input_df.columns]
            origin_input_df = scaled_input_df.copy(deep=True)
            scaled_input_df.columns = [f"scaled_input_{col}"  for col in scaled_input_df.columns]

            origin_input_df[df_info.numerical_cols] = df_info.scaler.inverse_transform(origin_input_df[df_info.numerical_cols])
            origin_input_df.columns = origin_columns

            final_df = final_df.join([scaled_input_df, origin_input_df])

            if not results[k][i]['cf'] is None:
                scaled_cf_df = results[k][i]['cf'].copy(deep=True)
                scaled_cf_df.loc[0, df_info.target_name] = df_info.target_label_encoder.inverse_transform([scaled_cf_df.loc[0, df_info.target_name]])[0]
                origin_cf_columns = [f"origin_cf_{col}"  for col in scaled_cf_df.columns]
                origin_cf_df = scaled_cf_df.copy(deep=True)
                scaled_cf_df.columns = [f"scaled_cf_{col}"  for col in scaled_cf_df.columns]

                origin_cf_df[df_info.numerical_cols] = df_info.scaler.inverse_transform(origin_cf_df[df_info.numerical_cols])
                origin_cf_df.columns = origin_cf_columns

                final_df = final_df.join([scaled_cf_df, origin_cf_df])

            # final_df = final_df.join([scaled_input_df, origin_input_df, scaled_cf_df, origin_cf_df])
            final_df['running_time'] = results[k][i]['running_time']
            final_df['Found'] = "Y" if not results[k][i]['cf'] is None else "N"
            final_df['ground_truth'] = results[k][i]['ground_truth'] 
            final_df['prediction'] = results[k][i]['prediction'] 

            all_data.append(final_df)

        result_dfs[k] = pd.concat(all_data)

    return result_dfs
