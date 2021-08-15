from utils.preprocessing import DfInfo
from alibi_cf.utils import get_cat_vars_dict
from alibi_cf import AlibiBinaryPredictWrapper
from alibi.explainers import CounterFactualProto
import numpy as np
from time import time
from utils.preprocessing import inverse_dummy
import pandas as pd

def get_cat_vars_info(cat_feature_names, train_df):
    cat_vars_idx_info = []

    for cat_col in cat_feature_names:
        num_unique_v = len([ col for col in train_df.columns if col.startswith(f"{cat_col}_")])
        first_index = min([ list(train_df.columns).index(col) for col in train_df.columns if col.startswith(f"{cat_col}_")])
        
        cat_vars_idx_info.append({
            "col": cat_col,
            "num_unique_v": num_unique_v,
            "first_index": first_index
        })

    cat_vars_ohe = {}
    
    for idx_info in cat_vars_idx_info:
        cat_vars_ohe[idx_info['first_index']] = idx_info['num_unique_v']

    return cat_vars_idx_info, cat_vars_ohe

def alibi_wrap_models(models):
    return {
        'dt': AlibiBinaryPredictWrapper(models['dt']),
        'rfc': AlibiBinaryPredictWrapper(models['rfc']),
        'nn': AlibiBinaryPredictWrapper(models['nn']),
    }

def get_proto_cfs(wrapped_models, feature_range, cat_vars_ohe, X_train):

    proto_cfs = {}

    for k in wrapped_models.keys():
        proto_cfs[k] = CounterFactualProto(
                                    wrapped_models[k].predict,
                                    X_train[0].reshape(1, -1).shape,
                                    cat_vars=cat_vars_ohe,
                                    feature_range=feature_range,
                                    max_iterations=500,
                                    ohe=True,
                                    )

        proto_cfs[k].fit(X_train)

    return proto_cfs



def generate_cf_proto_result(df_info: DfInfo, train_df, models, num_instances, num_cf_per_instance, X_train, X_test, y_test):
    cat_feature_names = [ col for col in df_info.categorical_cols if col != df_info.target_name ] 

    cat_vars_idx_info, cat_vars_ohe = get_cat_vars_info(cat_feature_names, train_df)

    wrapped_models = alibi_wrap_models(models)

    feature_range = (np.ones((1, len(df_info.feature_names))), np.zeros((1, len(df_info.feature_names))))

    profo_cfs = get_proto_cfs(wrapped_models, feature_range, cat_vars_ohe, X_train)

    results = {}
    for k in profo_cfs.keys():
        results[k] = []
        print(f"Finding counterfactual for {k}")
        for idx, instance in enumerate(X_test[0:num_instances]):
            print(f"instance {idx}")
            example = instance.reshape(1, -1)
            for num_cf in range(num_cf_per_instance):
                print(f"CF {num_cf}")
                start_t = time()
                exp = profo_cfs[k].explain(example)
                end_t = time ()
                running_time = end_t - start_t

                if k=='nn':
                    prediction = df_info.target_label_encoder.inverse_transform((models[k].predict(example)[0]> 0.5).astype(int))[0]
                else:
                    prediction = df_info.target_label_encoder.inverse_transform(models[k].predict(example))[0]

                if (not exp.cf is None) and (len(exp.cf) > 0):
                    print("Found CF")
                    if k == 'nn':
                        cf = inverse_dummy(pd.DataFrame(exp.cf['X'], columns=df_info.ohe_feature_names), df_info.all_cat_ohe_cols)
                        cf.loc[0, df_info.target_name] = df_info.target_label_encoder.inverse_transform([exp.cf['class']])[0]
                    else:
                        cf = inverse_dummy(pd.DataFrame(exp.cf['X'], columns=df_info.ohe_feature_names), df_info.all_cat_ohe_cols)
                        cf.loc[0, df_info.target_name] = df_info.target_label_encoder.inverse_transform([exp.cf['class']])[0]
                else:
                    print("CF not found")
                    cf = None

                input_df = inverse_dummy(pd.DataFrame(example, columns=df_info.ohe_feature_names), df_info.all_cat_ohe_cols)
                input_df.loc[0, df_info.target_name] = prediction

                results[k].append({
                    "input": input_df,
                    "cf": cf,
                    'exp': exp,
                    "running_time": running_time,
                    "ground_truth": df_info.target_label_encoder.inverse_transform([y_test[idx]])[0],
                    "prediction": prediction,
                })

    return results

def process_result(results, df_info):

    results_df = {}

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
                ## Comment this
                # scaled_cf_df.loc[0, target_name] = target_label_encoder.inverse_transform([scaled_cf_df.loc[0, target_name]])[0]
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

        results_df[k] = pd.concat(all_data)

    return results_df

    



    




