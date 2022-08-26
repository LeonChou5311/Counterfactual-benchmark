import numpy as np
import pandas as pd
from enum import Enum
from typing import List
from utils.preprocessing import DfInfo
from scipy.spatial import distance


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

    # should remove the target column first.
    
    input_df = kwargs['not_dummy_input']
    cf_df = kwargs['not_dummy_cf']

    input_array = np.array(input_df)
    cf_array = np.array(cf_df)

    return (input_array != cf_array).astype(int).sum(axis=1)

def get_realisitic(**kwargs):
    '''
    Checking if the numerical columns are in the range of [0, 1].
    '''
    df_info: DfInfo = kwargs['df_info']
    cf_num_array = np.array(kwargs['cf'][df_info.numerical_cols])
    return np.all(np.logical_and(cf_num_array >= 0, cf_num_array <= 1 ), axis=1)

def get_mad(**kwargs,):
    '''
    Get Mean Absolute Deviation Distance between input and cf. 
    '''

    eps = 1e-8

    input_df = kwargs['input']
    cf_df = kwargs['cf']
    df_info = kwargs['df_info']

    ohe_cat_cols = df_info.get_ohe_cat_cols()
    ohe_num_cols = df_info.get_ohe_num_cols()

    numerical_mads = df_info.get_numerical_mads()

    mad_df = pd.DataFrame({}, columns= df_info.ohe_feature_names)
    mad_df[ohe_cat_cols] = (input_df[ohe_cat_cols] != cf_df[ohe_cat_cols]).astype(int)
    for num_col in ohe_num_cols: 
        mad_df[num_col] = abs(cf_df[num_col] - input_df[num_col]) / (numerical_mads[num_col] + eps)

    if len(ohe_cat_cols) > 0 and len(ohe_num_cols) > 0:
        return (mad_df[ohe_num_cols].mean(axis=1) + mad_df[ohe_cat_cols].mean(axis=1)).tolist()
        # return mad_df.mean(axis=1).tolist()
        # return mad_df.sum(axis=1).tolist() # <=(weird, may be wrong) actually from (https://github.com/ADMAntwerp/CounterfactualBenchmark/blob/9dbf6a9e604ce1a2a0ddfb15025718f2e0effb0a/frameworks/LORE/distance_functions.py) 

    elif len(ohe_num_cols) > 0:
        return mad_df[ohe_num_cols].mean(axis=1).tolist()
    elif len(ohe_cat_cols) > 0:
        return mad_df[ohe_cat_cols].mean(axis=1).tolist()
    else:
        raise Exception("No columns provided for MAD.")

    # return (mad_df[ohe_num_cols].mean(axis=1) + mad_df[ohe_cat_cols].mean(axis=1)).tolist()


def get_mahalanobis(**kwargs,):
    '''
    Get Mahalanobis distance between input and cf.
    '''
    input_df = kwargs['input']
    cf_df = kwargs['cf']
    df_info = kwargs['df_info']

    VI_m = df_info.dummy_df[df_info.ohe_feature_names].cov().to_numpy()

    return [distance.mahalanobis(input_df[df_info.ohe_feature_names].iloc[i].to_numpy(),
                                cf_df[df_info.ohe_feature_names].iloc[i].to_numpy(),
                                VI_m) for i in range(len(input_df))]

class EvaluationMatrix(Enum):
    '''
    All evaluation function should be registed here.
    '''
    L1 = "L1"
    L2 = "L2"
    Sparsity = "Sparsity"
    Realistic = "Realistic"
    MAD = "MAD"
    Mahalanobis = "Mahalanobis"

evaluation_name_to_func = {
    # All evaluation function should be registed here as well
    EvaluationMatrix.L1: get_L1,
    EvaluationMatrix.L2: get_L2,
    EvaluationMatrix.Sparsity: get_sparsity,
    EvaluationMatrix.Realistic: get_realisitic,
    EvaluationMatrix.MAD: get_mad,
    EvaluationMatrix.Mahalanobis: get_mahalanobis,
}


'''
Util functions.
'''

def get_dummy_version(input_df: pd.DataFrame, df_info: DfInfo):
    '''
    Transform the categorical data to ohe format. (Better for calculating the distance)
    '''

    def get_string_dummy_value(x):
        if isinstance(x, float) and x==x:
            x = int(x)

        return str(x)

    number_of_instances = len(input_df)

    init_row = {}
    for k in df_info.ohe_feature_names:
        init_row[k] = 0

    init_df = pd.DataFrame([init_row]*number_of_instances,
                           columns=df_info.ohe_feature_names)

    for k, v in df_info.cat_to_ohe_cat.items():
        for ohe_f in v:
            init_df[ohe_f] = input_df[k].apply(
                lambda x: 1 if ohe_f.endswith(get_string_dummy_value(x)) else 0).tolist()

    for col in df_info.numerical_cols:
        init_df[col] = input_df[col].tolist()

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
        "not_dummy_input": get_type_instance(result_df, InstanceType.ScaledInput).drop(df_info.target_name, axis=1),
        "not_dummy_cf": get_type_instance(result_df, InstanceType.ScaledCf).drop(df_info.target_name, axis=1),
        "df_info": df_info,
    }


def get_evaluations(result_df: pd.DataFrame, df_info: DfInfo, matrix: List[EvaluationMatrix]):
    '''
    Perform evaluation on the result dataframe according to the matrix given.

    [result_df] -> data frame containing input query and its counterfactaul.
    [df_info] -> DfInfo instance containing all data information.
    [matrix] -> The evaluation matrix to perform on `result_df`.
    '''

    evaluation_df = result_df.copy(deep=True)

    ## Only perform evaluation on the row with found cf.
    found_idx = evaluation_df[evaluation_df['Found']=="Y"].index
    cf_found_eaval_df = evaluation_df.loc[found_idx].copy(deep=True)

    if len(cf_found_eaval_df) < 1:
        raise Exception("No counterfactuals found, can't provide any evaluation.")

    input_and_cf = prepare_evaluation_dict(cf_found_eaval_df, df_info)

    for m in matrix:
        cf_found_eaval_df[m.value] = evaluation_name_to_func[m](**input_and_cf)

    evaluation_df.loc[found_idx, cf_found_eaval_df.columns] = cf_found_eaval_df

    return evaluation_df
