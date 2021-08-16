from sklearn.preprocessing import LabelEncoder

import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

def get_columns_type(df):

    integer_features = list(df.select_dtypes(include=['int64']).columns)
    float_features = list(df.select_dtypes(include=['float64']).columns)
    string_features = list(df.select_dtypes(include=['object']).columns)
    columns_type = {
        'integer': integer_features,
        'float': float_features,
        'string': string_features,
    }

    numerical_cols = columns_type['integer'] + columns_type['float']
    categorical_cols = columns_type['string']

    return numerical_cols, categorical_cols, columns_type

def transform_to_dummy(df, columns):
    ## For nerual network, we feed in the one-hot encoded vector.
    for col in columns:
        df = pd.concat([df,pd.get_dummies(df[col], prefix=col)],axis=1)
        df.drop([col],axis=1, inplace=True)
    return df

def label_encode(df, columns, encoder_dict=None):
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

def label_encode(df, columns, encoder_dict=None):
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
    temp_df = df.copy(deep=True)
    for col in columns:
        encoder = encoder_dict[col]
        temp_df[col] = encoder.inverse_transform(temp_df[col])
    return temp_df

def remove_missing_values(df):
    # df = pd.DataFrame(df.to_dict())
    for col in df.columns:
        if '?' in list(df[col].unique()):
            ### Replace the missing value by the most common.
            df[col][df[col] == '?'] = df[col].value_counts().index[0]
            
    return df

def min_max_scale_numerical(df, numerical_cols):
    ## Scaling the numerical data.
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df, scaler

def inverse_dummy(dummy_df, all_cat_ohe_cols):
    not_dummy_df = dummy_df.copy(deep=True)
    for k in all_cat_ohe_cols.keys():
        not_dummy_df[k] = dummy_df[all_cat_ohe_cols[k]].idxmax(axis=1)
        not_dummy_df[k] = not_dummy_df[k].apply(lambda x: x.replace(f'{k}_',""))
        not_dummy_df.drop(all_cat_ohe_cols[k], axis=1, inplace=True)
    return not_dummy_df

def get_cat_ohe_info(dummy_df, categorical_cols, target_name):
    all_cat_ohe_cols = {}
    for c_col in categorical_cols:
        if c_col != target_name:
            all_cat_ohe_cols[c_col] = [ ohe_col for ohe_col in dummy_df.columns if ohe_col.startswith(c_col) and ohe_col != target_name]

    ohe_feature_names = [ col for col in dummy_df.columns if col != target_name]

    return all_cat_ohe_cols, ohe_feature_names


def preprocess_df(df_load_fn):

    ## Load df information
    df, feature_names, numerical_cols, categorical_cols, columns_type, target_name, possible_outcomes = df_load_fn()

    ## Apply MinMacScaler [0, 1]
    scaled_df, scaler = min_max_scale_numerical(df, numerical_cols)

    ## Get one-hot encoded features.
    dummy_df = pd.get_dummies(scaled_df, columns=  [ col for col in categorical_cols if col != target_name])

    ## Get one-hot encoded info
    all_cat_ohe_cols, ohe_feature_names = get_cat_ohe_info(dummy_df, categorical_cols, target_name)

    ## Preprocessing the label
    target_label_encoder = LabelEncoder()
    
    dummy_df[target_name] = target_label_encoder.fit_transform(dummy_df[target_name])

    ## Organise the order
    dummy_df= dummy_df[ohe_feature_names + [target_name]]

    return DfInfo(df, feature_names, numerical_cols, categorical_cols, columns_type, target_name, possible_outcomes, scaled_df, scaler, all_cat_ohe_cols, ohe_feature_names, target_label_encoder, dummy_df)


class DfInfo():
    def __init__(self,df, feature_names, numerical_cols, categorical_cols, columns_type, target_name, possible_outcomes, scaled_df, scaler, all_cat_ohe_cols, ohe_feature_names, target_label_encoder, dummy_df,):
        self.df = df
        self.feature_names = feature_names
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.columns_type = columns_type
        self.target_name = target_name
        self.possible_outcomes = possible_outcomes
        self.scaled_df = scaled_df
        self.scaler = scaler
        self.all_cat_ohe_cols = all_cat_ohe_cols
        self.ohe_feature_names = ohe_feature_names
        self.target_label_encoder = target_label_encoder
        self.dummy_df = dummy_df









