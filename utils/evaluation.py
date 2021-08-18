from typing import List
from utils.preprocessing import DfInfo
import pandas as pd
import numpy as np
from enum import Enum


class InstanceType(Enum):
    ScaledInput = "scaled_input_"
    ScaledCf = "scaled_cf_"
    OriginInput = "origin_input_"
    OriingCf = "origin_cf_"

'''
Evaluation Functions.
'''

def get_L2(**kwargs):
    input_array = np.array(kwargs['input'])
    cf_array = np.array(kwargs['cf'])

    return np.linalg.norm(input_array - cf_array, axis=1, ord=2)


def get_L1(**kwargs):
    input_array = np.array(kwargs['input'])
    cf_array = np.array(kwargs['cf'])

    return np.linalg.norm(input_array - cf_array, axis=1, ord=1)


def get_sparsity(**kwargs):
    input_array = np.array(kwargs['input'])
    cf_array = np.array(kwargs['cf'])

    return (input_array != cf_array).astype(int).sum(axis=1)

class EvaluationMatrix(Enum):
    '''
    All evaluation function should be registed here.
    '''
    L1 = "L1"
    L2 = "L2"
    Sparsity = "Sparsity"

evaluation_name_to_func = {
    # All evaluation function should be registed here as well
    EvaluationMatrix.L1: get_L1,
    EvaluationMatrix.L2: get_L2,
    EvaluationMatrix.Sparsity: get_sparsity,
}


'''
Util functions.
'''

def get_dummy_version(input_df: pd.DataFrame, df_info: DfInfo):
    '''
    Transform the categorical data to ohe format. (Better for calculating the distance)
    '''

    number_of_instances = len(input_df)

    init_row = {}
    for k in df_info.ohe_feature_names:
        init_row[k] = 0

    init_df = pd.DataFrame([init_row]*number_of_instances,
                           columns=df_info.ohe_feature_names)

    for k, v in df_info.cat_to_ohe_cat.items():
        for ohe_f in v:
            init_df[ohe_f] = input_df[k].apply(
                lambda x: 1 if ohe_f.endswith(x) else 0)

    for col in df_info.numerical_cols:
        init_df[col] = input_df[col]

    return init_df


def get_type_instance(df: pd.DataFrame, instance_type: InstanceType, with_original_name: bool = True):
    '''
    Get certain type of instance in the result data frame. Check `InstanceType` to know all types.
    '''

    df = df.copy(deep=True)
    return_df = df[[
        col for col in df.columns if col.startswith(instance_type.value)]]

    if with_original_name:
        return_df.columns = [col.replace(
            instance_type.value, "") for col in return_df.columns]

    return return_df


def prepare_evaluation_dict(result_df: pd.DataFrame, df_info: DfInfo):
    '''
    Prepare the information needed to perform evaluation.
    '''

    return {
        "input": get_dummy_version(get_type_instance(result_df, InstanceType.ScaledInput), df_info),
        "cf": get_dummy_version(get_type_instance(result_df, InstanceType.ScaledCf), df_info),
    }


def get_evaluations(result_df: pd.DataFrame, df_info: DfInfo, matrix: List[EvaluationMatrix]):
    '''
    Perform evaluation on the result dataframe according to the matrix given.

    [result_df] -> data frame containing input query and its counterfactaul.
    [df_info] -> DfInfo instance containing all data information.
    [matrix] -> The evaluation matrix to perform on `result_df`.
    '''

    evaluation_df = result_df.copy(deep=True)
    input_and_cf = prepare_evaluation_dict(evaluation_df, df_info)

    for m in matrix:
        evaluation_df[m.value] = evaluation_name_to_func[m](**input_and_cf)

    return evaluation_df
