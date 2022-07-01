from dataclasses import dataclass
from typing import Any, Dict, List
from sklearn.preprocessing import LabelEncoder
from itertools import chain

import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

def get_columns_type(df):
    '''
    Identify the column types to later classify them as categorical or numerical columns (features).
    '''
    integer_features = list(df.select_dtypes(include=['int64']).columns) +  list(df.select_dtypes(include=['int32']).columns)
    float_features = list(df.select_dtypes(include=['float64']).columns) + list(df.select_dtypes(include=['float32']).columns)
    string_features = list(df.select_dtypes(include=['object']).columns)
    columns_type = {
        'integer': integer_features,
        'float': float_features,
        'string': string_features,
    }

    numerical_cols = columns_type['integer'] + columns_type['float']
    categorical_cols = columns_type['string']

    return numerical_cols, categorical_cols, columns_type

def transform_to_dummy(df, categorical_cols):
    '''
    Tranform the categorical columns to ohe format.
    '''
    ## For nerual network, we feed in the one-hot encoded vector.
    for col in categorical_cols:
        df = pd.concat([df,pd.get_dummies(df[col], prefix=col)],axis=1)
        df.drop([col],axis=1, inplace=True)
    return df

def label_encode(df, columns, encoder_dict=None):
    '''
    (LORE implementation)
    Encode the categorical columns in ordirnal encoding format.
    LORE using it for categorical data, but we use ohe instead.
    '''

    df_temp = df.copy(deep=True)

    if encoder_dict:
        for col in columns:
            col_encoder = encoder_dict[col]
            df_temp[col] = col_encoder.transform(df_temp[col])
    else:
        encoder_dict = {}
        for col in columns:
            encoder = LabelEncoder()
            df_temp[col] = encoder.fit_transform(df_temp[col])
            encoder_dict[col] = encoder

    return df_temp, encoder_dict

def label_decode(df, columns, encoder_dict):
    '''
    Inverse process for `label_encode`.
    '''
    temp_df = df.copy(deep=True)
    for col in columns:
        encoder = encoder_dict[col]
        temp_df[col] = encoder.inverse_transform(temp_df[col])
    return temp_df

def remove_missing_values(df):
    '''
    Remove the rows with missing value in the dataframe.
    '''

    # df = pd.DataFrame(df.to_dict())
    for col in df.columns:
        if '?' in list(df[col].unique()):
            ### Replace the missing value by the most common.
            df[col][df[col] == '?'] = df[col].value_counts().index[0]
            
    return df

def min_max_scale_numerical(df, numerical_cols):
    '''
    Scale the numerical columns in the dataframe.
    '''

    ## Scaling the numerical data.
    scaled_df = df.copy(deep=True)
    scaler = MinMaxScaler()
    scaled_df[numerical_cols] = scaler.fit_transform(scaled_df[numerical_cols])
    return scaled_df, scaler

def inverse_dummy(dummy_df, cat_to_ohe_cat):
    '''
    Inverse the process of pd.get_dummy().
    [`cat_to_ohe_cat`] -> Dictionary `{"column_name": list("ohe_column_name")}`
    '''
    not_dummy_df = dummy_df.copy(deep=True)
    for k in cat_to_ohe_cat.keys():
        not_dummy_df[k] = dummy_df[cat_to_ohe_cat[k]].idxmax(axis=1)
        not_dummy_df[k] = not_dummy_df[k].apply(lambda x: x.replace(f'{k}_',""))
        not_dummy_df.drop(cat_to_ohe_cat[k], axis=1, inplace=True)
    return not_dummy_df

def inverse_scaling(scaled_df, df_info):
    result_df = scaled_df.copy(deep=True)

    result_df[df_info.numerical_cols] = df_info.scaler.inverse_transform(
                    result_df[df_info.numerical_cols])

    return result_df

def inverse_scaling_and_dummy(scaled_dummy_df, df_info):
    return inverse_scaling(inverse_dummy(scaled_dummy_df, df_info.cat_to_ohe_cat), df_info)

def get_cat_ohe_info(dummy_df, categorical_cols, target_name):
    '''
    Get ohe informatoin required for counterfactual generator (DiCE, Alibi) to recognise categorical features. 
    '''

    cat_to_ohe_cat = {}
    for c_col in categorical_cols:
        if c_col != target_name:
            cat_to_ohe_cat[c_col] = [ ohe_col for ohe_col in dummy_df.columns if ohe_col.startswith(c_col) and ohe_col != target_name]

    ohe_feature_names = [ col for col in dummy_df.columns if col != target_name]

    return cat_to_ohe_cat, ohe_feature_names


def preprocess_df(df_load_fn):
    '''
    Process the dataframe to extract necessary information.
    '''

    ## Load df information
    df, feature_names, numerical_cols, categorical_cols, columns_type, target_name, possible_outcomes = df_load_fn()

    ## Apply MinMacScaler [0, 1]
    scaled_df, scaler = min_max_scale_numerical(df, numerical_cols)

    ## Get one-hot encoded features.
    dummy_df = pd.get_dummies(scaled_df, columns=  [ col for col in categorical_cols if col != target_name])

    ## Get one-hot encoded info
    cat_to_ohe_cat, ohe_feature_names = get_cat_ohe_info(dummy_df, categorical_cols, target_name)

    ## Preprocessing the label
    target_label_encoder = LabelEncoder()
    
    dummy_df[target_name] = target_label_encoder.fit_transform(dummy_df[target_name])

    ## Organise the order
    dummy_df= dummy_df[ohe_feature_names + [target_name]]

    return DfInfo(df, feature_names, numerical_cols, categorical_cols, columns_type, target_name, possible_outcomes, scaled_df, scaler, cat_to_ohe_cat, ohe_feature_names, target_label_encoder, dummy_df)

@dataclass
class DfInfo:

    ## Original data frame
    df: pd.DataFrame 

    ## All feature names
    feature_names: List

    ###### `numerical_cols` and `categorical_cols` may contain column name of target.
    
    ## All numerical columns
    numerical_cols: List

    ## All categorical columns
    categorical_cols: List

    ## type of each columns
    columns_type: Dict

    ## Label(target) column name
    target_name: str

    ## Unique values in the target column.
    possible_outcomes: List 

    ## Dataframe with the numerical columns scaled by MinMaxScaler to [0, 1]
    scaled_df: pd.DataFrame

    ## MinMaxScaler to scale numerical columns.
    scaler: Any

    ## Dictionary {"categorical_col_name": "all of its ohe column names"}  
    cat_to_ohe_cat: Dict

    ## All feature names in ohe format
    ohe_feature_names: List

    ## LabelEncoder used to encoding the target column.
    target_label_encoder: Any

    ## Dataframe with scaled numerical features and dummy categorical features (The dataframe used for training).
    dummy_df: pd.DataFrame


    def get_ohe_cat_cols(self,):
        return list(chain(*[ v for v in self.cat_to_ohe_cat.values()]))

    def get_ohe_num_cols(self,):
        return self.numerical_cols

    def get_numerical_mads(self,):
        return self.scaled_df[self.numerical_cols].mad().to_dict()










