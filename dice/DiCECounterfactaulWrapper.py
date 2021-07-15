from copy import deepcopy
from time import time
from utils.print import print_block
import pandas as pd
import numpy as np


class DiCECounterfactaulWrapper(object):
    '''
    Wrapper class to generate DiCE cf
    '''

    def __init__(self, dice_explainer, feature_names):
        self.dice_explainer__ = dice_explainer
        self.feature_names = feature_names

    def run_counterfactual_print_result(self, case):
        return_case = deepcopy(case)
        # print_block("", "Finding counterfactuals...")
        input_data = np.array([case["original_vector"]])
        start_time = time()
        input_df = pd.DataFrame(input_data, columns=self.feature_names)
        self.input_df = input_df
        dice_exp = self.dice_explainer__.generate_counterfactuals(input_df, total_CFs=3, desired_class="opposite",
                                                                  # proximity_weight=1.5, diversity_weight=1.0
                                                                  )
        self.dice_exp = dice_exp
        end_time = time()
        time_took = end_time - start_time
        # print_block("Time Took", "%.3f sec" % (time_took))
        if len(dice_exp.final_cfs_df) == 0:
            # print_block("", "No counterfactaul found!")
            return_case['cf'] = [None] * input_data.shape[1]
        else:
            # counterfactual = scaler.inverse_transform(list())
            return_case['cf'] = list(dice_exp.final_cfs_df.iloc[0][:-1])

        return_case['time'] = time_took
        # self.print_counterfactual_results(case, counterfactual)
        return return_case

    def print_counterfactual_results(self, case, counterfactual):
        print_block("Prediction type", case["prediction_type"], mark_times=7)
        print_block("Black box prediction", case["predictions"], mark_times=3)
        print_block("Ground truth", case["ground_truth"], mark_times=5)
        print_block("Oringal input", pd.DataFrame(
            [case["original_vector"]], columns=self.feature_names), mark_times=60)
        print_block("Counterfactual", pd.DataFrame(
            counterfactual, columns=self.feature_names), mark_times=60)


