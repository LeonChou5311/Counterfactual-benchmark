import os

def save_result_as_csv(cf_alg_name, dataset_name, results_df):
    os.makedirs(f"./results/{cf_alg_name}_{dataset_name}", exist_ok=True)
    for df_k in results_df.keys():
        results_df[df_k].to_csv(f"./results/{cf_alg_name}_{dataset_name}/{cf_alg_name}_{dataset_name}_{df_k}_result.csv")
    