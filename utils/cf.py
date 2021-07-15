from utils.print import print_block
import pandas as pd
from utils.prediction import PredictionType


def generate_cf_for_all(packs, cf_func, feature_names):

    output_column_names = [f'orgin_{f}' for f in feature_names] + [
        f'cf_{f}' for f in feature_names] + ['time(sec)'] + ["prediction_type"]

    # Create an empty dataframe for appending data.
    result_df = pd.DataFrame({}, columns=output_column_names)

    # Loop through each predict type.
    for p_t in [PredictionType.TruePositive, PredictionType.TrueNegative, PredictionType.FalsePositive, PredictionType.FalseNegative]:
        print_block("", "Doing %s" % p_t.value)

        # Get the length, so we can through all the instance in this predict type.
        total_length = packs.get_len(p_t)

        # Loop through all the instance in this predict type.
        for i in range(total_length):
            print_block("Instance %d" % i, "Running...")

            # Get the result (including counterfactal and running time) from the cf_func.
            returned_case = cf_func(packs.get_instance(p_t, i))

            # Using the information from returned_case to create a dataframe (for appending to result_df).
            df_i = pd.DataFrame([
                returned_case["original_vector"] + returned_case['cf'] + [returned_case['time'], returned_case['prediction_type']]], columns=output_column_names)

            # appending the current result to the total result dataframe.
            result_df = result_df.append(df_i)

    return result_df
