def get_cat_vars_dict(df, categorical_cols, feature_names, target_name):
    cat_vars_dict = {}
    for col in [ col for col in categorical_cols if col != target_name]:
        cat_vars_dict[feature_names.index(col)] = len(df[col].unique())
    return cat_vars_dict