#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 09/02/2024

@author: Maurizio Ferrari Dacrema
"""


from Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
from Data_manager.DataPostprocessing_K_Cores import DataPostprocessing_K_Cores
from Data_manager.data_consistency_check import assert_disjoint_matrices, assert_implicit_data
from Evaluation.Evaluator import EvaluatorHoldout
from Data_manager import *
from Recommenders.Recommender_import_list import *
from Recommenders.DataIO import DataIO

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

    result_folder_path = "./{}/{}/hyperopt_{}/{}/".format("k_{}_cores".format(k_cores) if k_cores > 0 else "full",
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
    dataSplitter.load_data(save_folder_path=data_folder_path)

    if preprocessing == "implicit":
        _make_data_implicit(dataSplitter)

    model_folder_path = "./{}/{}/hyperopt_{}/{}/models/".format("k_{}_cores".format(k_cores) if k_cores > 0 else "full",
                                                                           preprocessing,
                                                                           split_type,
                                                                           dataset_reader._get_dataset_name())

    return dataSplitter, model_folder_path




if __name__ == '__main__':
    """
    This folder contains the result of hyperparameter optimization on 31 datasets and approximately 30 recommendation algorithms
    and variants, for a current total of 30+ years of CPU time.
    
    The datasets are split and preprocessed in various ways. The folder root structure is as follows:
    /data_filtering/preprocessing/data_split/
    
    - data filtering: 
            "full" is the whole dataset; 
            "k_5_cores" contains a densely connected subgraph with only users and items with at least 5 interactions. 
            "k_20_cores" contains a densely connected subgraph with only users and items with at least 20 interactions. 
                    Since removing users or items will reduce the number of iterations in the dataset, the process is repeated iteratively until all remaining users and item meet this constraint. Some sparse datasets do not have 5_cores or 20_cores because the process dones not converge and the filtering removes all data.
    
    - preprocessing:
            "original" refers to the data as in the original dataset
            "implicit" if the original dataset contains explicit data or other non binary data (eg. the number of times a slong was listened to), the data is made implicit.
                    Note that according to this categorization if the original dataset is implicit then it will be present only in the "original" folder.
    
    - data_split: 
            "hyperopt_leave_1_out", the validation and test data contain one random interaction of the user. 
                    If the user does not have enough interactions their profile is removed.
                    The suffix "only_warm_users" in the zip file contains the data refers to this behaviour.
            "hyperopt_random_holdout_80_10_10", the validation and test data contain 10% of the interactions sampled globally. Note that the interactions are not sampled holding out 10% of the interactions of each user, but 10% from the data regardless for the users they belong to.
                    Users that, after the global sampling, do not have interactions in the training data are removed. The suffix "only_warm_users" in the zip file contains the data refers to this behaviour.
    
    
    The optimization is performed with a Bayesian Search, optimizing NDCG@10. 
    For each algorithm and dataset split, the optimization uses 50 attempts, the first 16 used for random initialization. 
    Each search has a total maximum computational budget of 14 days, after which the search is stopped regardless for how many attempts were performed.
    Note that this may allow very few attempts (even 1-3) for highly computationally expensive algorithms on large datasets.
    
    In the /data_filtering/preprocessing/data_split/ folder, there are several subfolders, one for each dataset.
    In each dataset folder there are two subfolders:
    - data: contains the actual train, validation, test data split
    - models: contains the results of the hyperparameter search. Each recommender algorithm has a "*_metadata.zip" file.
              The metadata file can be loaded with the DataIO class and contains a lot of information:
              
              "algorithm_name_search": string, the name of the hyperparameter search method (BayesianSearch or SearchSingleCase for those that do not have hyperparameters),
              "algorithm_name_recommender": string, the name of the recommender algorithm,
              "metric_to_optimize": string, the name of the metric that is being optimized in this search,
              "cutoff_to_optimize": int, the name of the recommendation list length that is being optimized in this search,
              "exception_list": list, contains a component for each of the 50 hyperparameter configurations. Each component can be either None or contain the string of the exception that was raised,

              "hyperparameters_df": DataFrame, contains the hyperparameters that are explored, the rows are the iteration/attempts and the columns the hyperparameter names,
              "hyperparameters_best": dictionary, contains the optimal hyperparameter values,
              "hyperparameters_best_index": int, contains the index of the attempt that resulted in the optimal hyperparameter values (can be used to acces the dataframes)

              "result_on_validation_df": DataFrame, contains the evaluation results on the validation data for each of the 50 attempts. Note that it contains multiple cutoff lengths.
              "result_on_validation_best": dictionary, contains the evaluation results on the validation data associated to the best hyperparameter values,
              "result_on_test_df": DataFrame, contains the evaluation results on the test data for each of the 50 attempts. Usually the test results are present only if that hyperparameter configuration was better than the best one found so far.
              "result_on_test_best": dictionary, contains the evaluation results on the test data associated to the best hyperparameter values,
              "result_on_earlystopping_df": DataFrame, contains the evaluation results on the validation data computed during the model training and for earlystopping. Can be used to study the model convergence.

              "time_df": DataFrame, contains the time required for the training of the model as well as for the evaluation on the validation and test data.

              "time_on_train_total": float (seconds), the total time required to train the model across all the successful attempts. The failed attempts are excluded. Reasons for failed attempts are: exceptions raised, early termination of the search because the 14 days computational budget wa exhausted.
              "time_on_train_avg": float (seconds), average time required to train the model across all the successful attempts,

              "time_on_validation_total": float (seconds), total time required to evaluate the model on validation data across all the successful attempts,
              "time_on_validation_avg": float (seconds), average time required to evaluate the model on validation data across all the successful attempts,

              "time_on_test_total": float (seconds), total time required to evaluate the model on test data across all the successful attempts,
              "time_on_test_avg": float (seconds), average time required to evaluate the model on test data across all the successful attempts,

              "result_on_last": DataFrame, contains the results of the evaluation on the test data of the final model (called "last"), trained on the union of training and validation data using the best hyperparameters found.
              "time_on_last_df": DataFrame, contains the time required for the training of the model as well as for the evaluation on the validation and test data.   

    """

    k_cores = 0
    split_type = "random_holdout_80_10_10"
    preprocessing = "original"
    dataset_class = CiteULike_aReader
    cutoff_to_optimize = 10
    cutoff_list = [5, 10, 20, 30, 40, 50, 100]

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
    recommender_instance = P3alphaRecommender(URM_train_last_test)
    search_metadata = dataIO.load_data(P3alphaRecommender.RECOMMENDER_NAME + "_metadata")
    optimal_hyperparams = search_metadata["hyperparameters_best"]

    # Fit model with optimal hyperparameters
    recommender_instance.fit(**optimal_hyperparams)
    result_df, result_str = evaluator_test.evaluateRecommender(recommender_instance)
    print(result_str)

    # Load results on test data as a dataframe
    result_on_test_last = search_metadata['result_on_last']





