import pandas as pd

from utils.preprocessing import get_columns_type, transform_to_dummy, label_encode, remove_missing_values


def load_adult_df():
    ##### Pre-defined #####
    target_name = 'class'

    df = pd.read_csv('./datasets/adult.csv',
                     delimiter=',', skipinitialspace=True)

    del df['fnlwgt']
    del df['education-num']

    feature_names = [col for col in df.columns if col != target_name]

    df = remove_missing_values(df)

    possible_outcomes = list(df[target_name].unique())

    numerical_cols, categorical_cols, columns_type = get_columns_type(df)

    return df, feature_names, numerical_cols, categorical_cols, columns_type, target_name, possible_outcomes
