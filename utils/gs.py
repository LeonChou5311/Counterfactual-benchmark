import pandas as pd
from time import time
from utils.preprocessing import DfInfo
import numpy as np

import growingspheres.counterfactuals as cf


class GSNNWrapper:
    """
    Wrapper for NN
    """

    def __init__(self, model) -> None:
        self.model = model

    def predict(self, x):
        return (
            (self.model.predict(x.astype(float)) > 0.5)
            .astype(int)
            .reshape(
                -1,
            )
        )


def generate_gs_result(
    df_info: DfInfo,
    test_df: pd.DataFrame,
    models,
    num_instances,
    num_cf_per_instance,
    n_in_layer=2000,
):

    wrapped_models = {
        "dt": models["dt"],
        "rfc": models["rfc"],
        "nn": GSNNWrapper(models["nn"]),
    }

    results = {}

    instance_cfs = {}

    for k in wrapped_models.keys():
        results[k] = []
        print(f"Finding counterfactual for {k}")
        for idx, instance in enumerate(
            df_info.scaled_df.iloc[test_df[0:num_instances].index].iloc
        ):
            print(f"instance {idx}")
            for num_cf in range(num_cf_per_instance):
                print(f"CF {num_cf}")
                start_t = time()
                input_query = pd.DataFrame([instance.to_dict()])
                ground_truth = input_query[df_info.target_name][0]
                obs = np.array(input_query.iloc[0][df_info.feature_names])
                instance_cf = cf.CounterfactualExplanation(
                    obs,  # .astype('float64'),
                    wrapped_models[k].predict,
                    method="GS",
                )
                instance_cf.fit(
                    n_in_layer=n_in_layer,
                    first_radius=0.1,
                    dicrease_radius=10,
                    sparse=True,
                    verbose=True,
                )
                ###
                instance_cfs[k] = instance_cf
                ###
                prediction = df_info.target_label_encoder.inverse_transform(
                    wrapped_models[k].predict(obs.reshape(1, -1))
                )[0]
                cf_df = pd.DataFrame([instance_cf.enemy], columns=df_info.feature_names)
                cf_df.loc[
                    0, df_info.target_name
                ] = df_info.target_label_encoder.inverse_transform(
                    wrapped_models[k].predict(instance_cf.enemy.reshape(1, -1))
                )[
                    0
                ]

                end_t = time()
                running_time = end_t - start_t
                results[k].append(
                    {
                        "input": input_query,
                        "cf": cf_df,
                        "running_time": running_time,
                        "ground_truth": ground_truth,
                        "prediction": prediction,
                    }
                )

    return results, instance_cfs


def process_results(df_info: DfInfo, results):
    result_dfs = {}

    for k in results.keys():

        all_data = []

        for i in range(len(results[k])):
            final_df = pd.DataFrame([{}])

            # Inverse the scaling process to get the original data for input.
            scaled_input_df = results[k][i]["input"].copy(deep=True)
            origin_columns = [f"origin_input_{col}" for col in scaled_input_df.columns]
            origin_input_df = scaled_input_df.copy(deep=True)
            scaled_input_df.columns = [
                f"scaled_input_{col}" for col in scaled_input_df.columns
            ]

            origin_input_df[df_info.numerical_cols] = df_info.scaler.inverse_transform(
                origin_input_df[df_info.numerical_cols]
            )
            origin_input_df.columns = origin_columns

            final_df = final_df.join([scaled_input_df, origin_input_df])

            # If counterfactaul found, inverse the scaling process to get the original data for cf.
            if not results[k][i]["cf"] is None:
                scaled_cf_df = results[k][i]["cf"].copy(deep=True)
                # scaled_cf_df.loc[0, df_info.target_name] = df_info.target_label_encoder.inverse_transform([scaled_cf_df.loc[0, df_info.target_name]])[0]
                origin_cf_columns = [f"origin_cf_{col}" for col in scaled_cf_df.columns]
                origin_cf_df = scaled_cf_df.copy(deep=True)
                scaled_cf_df.columns = [
                    f"scaled_cf_{col}" for col in scaled_cf_df.columns
                ]

                origin_cf_df[df_info.numerical_cols] = df_info.scaler.inverse_transform(
                    origin_cf_df[df_info.numerical_cols]
                )
                origin_cf_df.columns = origin_cf_columns

                final_df = final_df.join([scaled_cf_df, origin_cf_df])

            # Record additional information.
            final_df["running_time"] = results[k][i]["running_time"]
            final_df["Found"] = "Y" if not results[k][i]["cf"] is None else "N"
            final_df["ground_truth"] = results[k][i]["ground_truth"]
            final_df["prediction"] = results[k][i]["prediction"]

            all_data.append(final_df)

        result_dfs[k] = pd.concat(all_data)

    return result_dfs
