from sklearn.preprocessing import LabelEncoder

import pandas as pd

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