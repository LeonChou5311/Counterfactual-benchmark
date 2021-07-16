
import numpy as np
from time import time
from copy import deepcopy
import pandas as pd
from utils.print import print_block



class AlibiCounterfactaulWrapper(object):
  '''
  Wrapper class to generate alibi cf
  '''
  def __init__(self, counterfactual_proto, watcher_counterfactual, scaler, feature_names):
    self.counterfactual_proto__ = counterfactual_proto
    self.watcher_counterfactual__ = watcher_counterfactual
    self.scaler = scaler
    self.feature_names = feature_names

    
  def counterfactual_proto_explain(self, case):
    return self.run_counterfactual_print_result(case, self.counterfactual_proto__)

  def watcher_counterfactual_explain(self, case):
    return self.run_counterfactual_print_result(case, self.watcher_counterfactual__)
  
  def run_counterfactual_print_result(self, case, cf):
    return_case = deepcopy(case)
    # print_block("", "Finding counterfactuals...")
    input_data = np.array([case["scaled_vector"]])
    start_time = time()
    explanation = cf.explain(input_data)
    end_time = time()
    time_took = end_time - start_time
    # print_block("Time Took", "%.3f sec" % (time_took))
    if explanation.cf == None:
      # print_block("", "No counterfactaul found!")
      return_case['cf'] = [None] * input_data.shape[1]
    else:  
      counterfactual = self.scaler.inverse_transform(explanation.cf['X'])
      return_case['cf'] = counterfactual.tolist()[0]

    return_case['time'] = time_took
    # self.print_counterfactual_results(case, counterfactual)
    return return_case

  def print_counterfactual_results(self, case, counterfactual):
    print_block("Prediction type", case["prediction_type"], mark_times=7)
    print_block("Black box prediction", case["predictions"], mark_times=3)
    print_block("Ground truth", case["ground_truth"], mark_times= 5)
    print_block("Oringal input", pd.DataFrame([case["original_vector"]], columns=self.feature_names),mark_times = 60)
    print_block("Counterfactual", pd.DataFrame(counterfactual, columns=self.feature_names), mark_times=60)