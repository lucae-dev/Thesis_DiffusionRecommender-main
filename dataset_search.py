import ast
from Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
from Data_manager.DataPostprocessing_K_Cores import DataPostprocessing_K_Cores
from Data_manager.data_consistency_check import assert_disjoint_matrices, assert_implicit_data
from Evaluation.Evaluator import EvaluatorHoldout
from Data_manager import *
from Data_manager.Movielens.Movielens1MReader import Movielens1MReader
from Diffusion.MultiBlockSimilarityAttentionDiffusionRecommender import MultiBlockSimilarityAttentionDiffusionRecommender
from Recommenders.DataIO import DataIO
import optuna
import numpy as np
import pandas as pd
import os
from Diffusion.MultiBlockAttentionDiffusionRecommenderSimilarity import MultiBlockAttentionDiffusionRecommenderInfSimilarity, MultiBlockAttentionDiffusionRecommenderSimilarity
from Diffusion.MultiBlockAttentionDiffusionRecommender import MultiBlockAttentionDiffusionRecommenderInf
from Diffusion.MultiBlockSimilarityAttentionDiffusionRecommender import MultiBlockSimilarityAttentionDiffusionRecommender
from Diffusion.MultiBlockWSimilarityAttentionDiffusionRecommender import WSAD_Recommender
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



def configure_fit_parameters(model, epochs, batch_size, embeddings_dim, heads, attention_blocks, d_ff, l2_reg, learning_rate, noise_timesteps, inference_timesteps, start_beta, end_beta, similarity_weight):
    # Initialize the parameter dictionary with mandatory and always applicable parameters
    params = {
        'epochs': epochs,
        'batch_size': batch_size,
        'embeddings_dim': embeddings_dim,
        'heads': heads,
        'attention_blocks': attention_blocks,
        'd_ff': d_ff,
        'l2_reg': l2_reg,
        'learning_rate': learning_rate,
        'noise_timesteps': noise_timesteps,
        'inference_timesteps': inference_timesteps,
        'start_beta': start_beta,
        'end_beta': end_beta
    }
    model = get_model()
    if model == MultiBlockSimilarityAttentionDiffusionRecommender or model == WSAD_Recommender:
        params['similarity_weight'] = similarity_weight

    return params




def objective(trial):

    cutoff = 10
    metric = 'NDCG'
    directory_path = './Self-Attention/OptunaResults/Dataset/' + (str(k_cores) if k_cores > 0 else "full") + '/' + dataset_class()._get_dataset_name()

    model = get_model()

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory {directory_path} created.")

    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512]) # , 1024]) # Movielens100k has only 943 users!!
    embeddings_dim = trial.suggest_categorical('embeddings_dim', [64, 128, 256, 512, 1024])
    heads = trial.suggest_categorical('heads', [1, 4, 8, 16])
    attention_blocks = trial.suggest_categorical('attention_blocks', [1])
    d_ff = trial.suggest_categorical('d_ff', [1024, 2048, 4096])
    epochs = trial.suggest_int('epochs', 20, 500)
    l2_reg = trial.suggest_loguniform('l2_reg', 1e-5, 1e-3)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    noise_timesteps = trial.suggest_int('noise_timesteps', 3, 80)
    inference_timesteps = trial.suggest_int('inference_timesteps', 2, noise_timesteps-1)
    start_beta = trial.suggest_float('start_beta', 0.00001, 0.001)
    end_beta = trial.suggest_float('end_beta', 0.01, 0.2)
    similarity_weight = trial.suggest_float('similiraty_weight', 0.1, 2)


    # Initialize and train the recommender

    diffusion_model = model(URM_train = URM_train, verbose = False, use_gpu = True)

    fit_param = configure_fit_parameters(
                      model = diffusion_model,
                      epochs=epochs,
                      batch_size=batch_size,
                      embeddings_dim=embeddings_dim,
                      heads=heads,
                      attention_blocks = attention_blocks,
                      d_ff = d_ff,
                      l2_reg=l2_reg,
                      learning_rate=learning_rate,
                      noise_timesteps = noise_timesteps,
                      inference_timesteps = inference_timesteps,
                      start_beta = start_beta,
                      end_beta = end_beta,
                      similarity_weight=similarity_weight
    )

    diffusion_model.fit(**fit_param)

    result_df, _ = evaluator_validation.evaluateRecommender(diffusion_model)
    hyperparams = {
    'batch_size': batch_size,
    'embeddings_dim': embeddings_dim,
    'heads': heads,
    'attention_blocks': attention_blocks,
    'd_ff': d_ff,
    'epochs': epochs,
    'l2_reg': l2_reg,
    'learning_rate': learning_rate,
    'noise_timesteps': noise_timesteps,
    'inference_timesteps': inference_timesteps,
    'start_beta': start_beta,
    'end_beta': end_beta}

    if model == MultiBlockSimilarityAttentionDiffusionRecommender or model == WSAD_Recommender:
        hyperparams['similarity_weight'] = similarity_weight

    result_df['hyperparams'] = str(hyperparams)

    filename = directory_path + '/' + diffusion_model.RECOMMENDER_NAME + ".csv"
    print(str(filename))
    # Check if file exists
    if os.path.isfile(filename):
        # If it exists, append without writing the header
        pd.DataFrame(result_df.loc[cutoff]).transpose().to_csv(filename, mode='a', header=False, index=False)
    else:
        # If it doesn't exist, create it, write the header
        pd.DataFrame(result_df.loc[cutoff]).transpose().to_csv(filename, mode='w', header=True, index=False)

    return result_df.loc[cutoff][metric]


def get_model():
    model_type = os.getenv("MODEL_TYPE")  
    
    if model_type == "InfSimilarity":
        model = MultiBlockAttentionDiffusionRecommenderInfSimilarity
    elif model_type == "Similarity":
        model = MultiBlockAttentionDiffusionRecommenderSimilarity
    elif model_type == "ADPR":
        model = MultiBlockAttentionDiffusionRecommenderInf
    elif model_type == "SAD":
        model = MultiBlockSimilarityAttentionDiffusionRecommender
    elif model_type == "WSAD":
        model = WSAD_Recommender
    else:
        raise ValueError("Unsupported model type specified.")
    
    return model

def get_dataset():
    dataset_class = os.getenv("DATASET") 
    
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

def should_save_on_remote_db():
    return os.getenv("SAVE", "False").lower() == "true"

if __name__ == '__main__':

    model = get_model()

    k_cores = 0
    split_type = "random_holdout_80_10_10"
    preprocessing = "implicit"
    dataset_class = get_dataset()
    cutoff_to_optimize = 10
    cutoff_list = [10]

    directory_path = './Self-Attention/OptunaResults/Dataset/' + (str(k_cores) if k_cores > 0 else "full") + '/' + dataset_class()._get_dataset_name()

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

    dataIO = DataIO(folder_path=model_folder_path)
    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list = cutoff_list)
    evaluator_validation_earlystopping = EvaluatorHoldout(URM_validation, cutoff_list = [cutoff_to_optimize])
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list = cutoff_list)


    # Load optimal hyperparams
    recommender_instance = model(URM_train_last_test)
    #search_metadata = {'batch_size': 256, 'embeddings_dim': 512, 'heads': 1, 'attention_blocks': 3, 'd_ff': 4096, 'epochs': 336, 'l2_reg': 0.0007440687899631993, 'learning_rate': 0.00036441971846935237, 'noise_timesteps': 81, 'inference_timesteps': 6, 'start_beta': 0.0006551779118897007, 'end_beta': 0.01539082762973255}# dataIO.load_data(P3alphaRecommender.RECOMMENDER_NAME + "_metadata")
    #optimal_hyperparams = search_metadata# search_metadata["hyperparameters_best"]
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, show_progress_bar=True)


    directory_path = './Self-Attention/OptunaResults/Dataset/' + (str(k_cores) if k_cores > 0 else "full") + '/' + dataset_class()._get_dataset_name()
    filename = directory_path + '/' + recommender_instance.RECOMMENDER_NAME + ".csv"

    df =  pd.read_csv(filename)
    optimal_hyperparams_str = df.loc[df['NDCG'].idxmax(), 'hyperparams']
    optimal_hyperparams = ast.literal_eval(optimal_hyperparams_str)
    print(optimal_hyperparams)
    # Fit model with optimal hyperparameters
    recommender_instance.fit(**optimal_hyperparams)
    result_df, result_str = evaluator_test.evaluateRecommender(recommender_instance)
    print(result_str)

    result_df['hyperparams'] = optimal_hyperparams_str
    result_df['model'] = recommender_instance.RECOMMENDER_NAME

    recommender_instance_name = "Multi-Block" + recommender_instance.RECOMMENDER_NAME 

    experiment_table_name = recommender_instance_name + '_' + dataset_class()._get_dataset_name() + 'experiment'
    result_table_name = dataset_class()._get_dataset_name() + 'best_result'

    if should_save_on_remote_db():
        save_to_db(df, experiment_table_name)
        save_to_db(result_df, result_table_name)
        
    print("fine esperimento!")





