import ast
import json

from Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
from Data_manager.DataPostprocessing_K_Cores import DataPostprocessing_K_Cores
from Data_manager.data_consistency_check import assert_disjoint_matrices, assert_implicit_data
from Evaluation.Evaluator import EvaluatorHoldout
from Data_manager import *
from Data_manager.Movielens.Movielens1MReader import Movielens1MReader
import numpy as np
import pandas as pd
import os
from Diffusion.MultiBlockAttentionDiffusionRecommenderSimilarity import MultiBlockAttentionDiffusionRecommenderInfSimilarity, MultiBlockAttentionDiffusionRecommenderSimilarity
from Diffusion.MultiBlockAttentionDiffusionRecommender import MultiBlockAttentionDiffusionRecommenderInf
from Diffusion.MultiBlockSimilarityAttentionDiffusionRecommender import SAD
from Diffusion.MultiBlockWSimilarityAttentionDiffusionRecommender import WSAD_Recommender
from Diffusion.LSSAD import LSSAD
import psycopg2
from psycopg2 import sql




def _make_data_implicit(dataSplitter):

    dataSplitter.SPLIT_URM_DICT["URM_train"].data = np.ones_like(dataSplitter.SPLIT_URM_DICT["URM_train"].data)
    dataSplitter.SPLIT_URM_DICT["URM_validation"].data = np.ones_like(dataSplitter.SPLIT_URM_DICT["URM_validation"].data)
    dataSplitter.SPLIT_URM_DICT["URM_test"].data = np.ones_like(dataSplitter.SPLIT_URM_DICT["URM_test"].data)

    URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()

    assert_disjoint_matrices([URM_train, URM_validation, URM_test])


def load_data(dataset_class, split_type, preprocessing, k_cores):

    from Data_manager.DataSplitter_Holdout import DataSplitter_Holdout

    dataset_reader = dataset_class()

    if k_cores > 0:
        dataset_reader = DataPostprocessing_K_Cores(dataset_reader, k_cores_value = k_cores)

    result_folder_path = "./Hyperparameter_databases/hyperparameter_database_2024_02/{}/{}/hyperopt_{}/{}/".format("k_{}_cores".format(k_cores) if k_cores > 0 else "full",
                                                                                                                                                    "original",
                                                                                                                                                    split_type,
                                                                                                                                                    dataset_reader._get_dataset_name())

    if split_type == "random_holdout_80_10_10":
        dataSplitter = DataSplitter_Holdout(dataset_reader, user_wise=False, split_interaction_quota_list=[80, 10, 10], forbid_new_split=True)

    elif split_type == "leave_1_out":
        dataSplitter = DataSplitter_leave_k_out(dataset_reader, k_out_value=1, use_validation_set=True, leave_random_out=True, forbid_new_split=True)
    else:
        raise ValueError

    data_folder_path = result_folder_path + "data/"
    print(data_folder_path)
    dataSplitter.load_data(save_folder_path=data_folder_path)

    if preprocessing == "implicit":
        _make_data_implicit(dataSplitter)

    model_folder_path = "./Hyperparameter_databases/hyperparameter_database_2024_02/{}/{}/hyperopt_{}/{}/models/".format("k_{}_cores".format(k_cores) if k_cores > 0 else "full",
                                                                           preprocessing,
                                                                           split_type,
                                                                           dataset_reader._get_dataset_name())
    return dataSplitter, model_folder_path

def get_model(model_type):    
    if model_type == "S_SAD_x0":
        model = MultiBlockAttentionDiffusionRecommenderInfSimilarity
    elif model_type == "LS_SAD_x0":
        model = LSSAD
    elif model_type == "DO NOT USE":
        model = MultiBlockAttentionDiffusionRecommenderSimilarity
    elif model_type == "ADPR_x0":
        model = MultiBlockAttentionDiffusionRecommenderInf
    elif model_type == "MB_ADPR_x0":
        model = MultiBlockAttentionDiffusionRecommenderInf
    elif model_type == "SAD_x0":
        model = SAD
    elif model_type == "WSAD":
        model = WSAD_Recommender
    else:
        model = None
    
    return model

def get_dataset(dataset_class):    
    if dataset_class == "Movielens1M":
        dataset_reader = Movielens1MReader
    elif dataset_class == "FilmTrust":
        dataset_reader = FilmTrustReader
    elif dataset_class == "Frappe":
        dataset_reader = FrappeReader
    elif dataset_class == "LastFMHetrec2011":
        dataset_reader = LastFMHetrec2011Reader

    else:
        raise ValueError("Unsupported dataset specified.")
    
    return dataset_reader

def get_connection():
    db_url = "postgresql://postgres:TGBCFSFxiLVUyNZInoIAfClDtkrTwZau@monorail.proxy.rlwy.net:18855/railway"
    
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()
    return conn, cursor

def close_connection(conn, cursor):
    conn.commit()
    cursor.close()
    conn.close()




    
def fetch_rows_from_db(table, models):
    conn, cursor = get_connection()
    cursor.execute(f"SELECT * FROM {table} WHERE model = ANY(%s)", (models,))
    columns = [col[0] for col in cursor.description]
    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    close_connection(conn, cursor)
    return rows

def save_to_db(df, table_name):
    conn, cursor = get_connection()
    
    column_str = ', '.join([f"{col} TEXT" if df[col].dtype == 'object' else f"{col} REAL" for col in df.columns])
    create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} (id SERIAL PRIMARY KEY, {column_str})"
    cursor.execute(create_table_query)

    columns = ', '.join(df.columns)
    placeholders = ', '.join(['%s'] * len(df.columns))
    sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

    # Convert DataFrame to list of tuples
    data_tuples = list(df.itertuples(index=False, name=None))

    # Execute the command
    cursor.executemany(sql, data_tuples)

    close_connection(conn, cursor)


def load_data_splitted(dataset_class):
    k_cores = 0
    split_type = "random_holdout_80_10_10"
    preprocessing = "implicit"
    cutoff_to_optimize = 10
    cutoff_list = [10]

    directory_path = './Inference_Analysis/Dataset/' + dataset_class()._get_dataset_name()

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f" !!!! Directory {directory_path} created.")

    dataSplitter, model_folder_path = load_data(dataset_class, split_type, preprocessing, k_cores)


    URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()
    URM_train_last_test = URM_train + URM_validation

    # Ensure disjoint test-train split
    assert_disjoint_matrices([URM_train, URM_validation, URM_test])

    # Ensure implicit data
    if preprocessing == "implicit":
        assert_implicit_data([URM_train, URM_validation, URM_test, URM_train_last_test])

    return URM_train, URM_validation, URM_test



if __name__ == '__main__':
    cutoff_to_optimize = 10
    cutoff_list = [10]
    # Define a mapping of model strings to recommender instances
    model_mapping = {
        'ADPR_x0': 'ADPR',
        'MB_ADPR_x0': 'ADPR',
        'SAD_x0': 'SAD',
        'S-SAD_x0': 'S-SAD',
        'LS_SAD_x0': 'LS_SAD'
    }

    tables_mapping = {
        'movielens1mbest_result': 'Movielens1M',
        'frappebest_result': 'Frappe',
        'filmtrustbest_result': 'FilmTrust',
        'lastfmhetrec2011best_result' : 'LastFMHetrec2011'
    }

    # Define your tables and models
    tables = [
        'movielens1mbest_result',
        'frappebest_result',
        'filmtrustbest_result',
        'lastfmhetrec2011best_result'
    ]
    models = ['ADPR_x0', 'MB_ADPR_x0', 'SAD_x0', 'S-SAD_x0', 'LS_SAD_x0']

    # Iterate over each table
    for table in tables:
        print(f"Processin table: {table}")
        URM_train, URM_validation, URM_test = load_data_splitted(get_dataset(tables_mapping[table]))
        evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list = cutoff_list)
        evaluator_validation_earlystopping = EvaluatorHoldout(URM_validation, cutoff_list = [cutoff_to_optimize])
        evaluator_test = EvaluatorHoldout(URM_test, cutoff_list = cutoff_list)

        rows = fetch_rows_from_db(table, models)

        for row in rows:
            model_name = row['model']
            hyperparams = row['hyperparams'] 
            # Parse hyperparams if they are in string format
            if isinstance(hyperparams, str):
                hyperparams = ast.literal_eval(hyperparams)
                
                
            # Fit the model with hyperparams (you need to implement this part)
            model = None
            model = get_model(model_name)
            recommender_instance = model(URM_train+URM_validation)

            print(f"Processing Model: {model_name}")
            
            # Fit the model
            if model:    
                hyperparams['noise_timesteps'] = 102
                recommender_instance.fit(**hyperparams)
                for inference_timestamp in range(0, 105, 5):
                    print("Processing inference timestamp: " + str(inference_timestamp) + "/100")
                    if inference_timestamp == 0:
                        inference_timestamp = 1
                    recommender_instance.set_inference_timesteps(inference_timestamp)
                    result_df, result_str = evaluator_test.evaluateRecommender(recommender_instance)
                    result_df['dataset'] = tables_mapping[table]
                    result_df['model'] = model_name
                    result_df['inference_timestamp'] = inference_timestamp
                    result_df['hyperparams'] = str(hyperparams)
                    save_to_db(result_df, "inference_timestep_analysis")
