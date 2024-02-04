#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19/06/2020

@author: Maurizio Ferrari Dacrema
"""


from argparse import ArgumentParser
from Recommenders.Recommender_import_list import *
from Data_manager import *

import HyperparameterTuning.functions_for_parallel_dataset as functions_for_parallel_dataset
import HyperparameterTuning.functions_for_parallel_model as functions_for_parallel_model
import mkl

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-b', '--baseline_tune',        help='Baseline hyperparameter search', type=bool, default=True)
    parser.add_argument('-p', '--print_results',        help='Print results', type=bool, default=True)

    input_flags = parser.parse_args()
    print(input_flags)

    KNN_similarity_to_report_list = ['cosine', 'dice', 'jaccard', 'asymmetric', 'tversky', 'euclidean']

    mkl.set_num_threads(4)
    # mkl.get_max_threads()

    # Leave 1 out
    dataset_list = [
        # AmazonAutomotiveReader,
        # AmazonBooksReader,
        # AmazonElectronicsReader,
        # AmazonMoviesTVReader,
        # AmazonMusicReader,
        # AmazonMusicalInstrumentsReader,
        # BookCrossingReader,
        # BrightkiteReader,
        # CiaoReader,
        # CiteULike_aReader,
        # CiteULike_tReader,
        # ContentWiseImpressionsReader,
        # DeliciousHetrec2011Reader,
        # EpinionsReader,
        # FilmTrustReader,
        # FrappeReader,
        # GowallaReader,
        # JesterJokesReader,
        # LastFMHetrec2011Reader,
        # MillionSongDatasetTasteReader,
        Movielens100KReader,
        # Movielens1MReader,
        # Movielens10MReader,
        # Movielens20MReader,
        # Movielens25MReader,
        # MovielensHetrec2011Reader,
        # NetflixPrizeReader,
        # PinterestReader,
        # TafengReader,
        # TheMoviesDatasetReader,
        # ThirtyMusicReader,
        # TVAudienceReader,
        # XingChallenge2016Reader,
        # XingChallenge2017Reader,
        # YelpReader,
    ]


    recommender_class_list = [
        Random,
        TopPop,
        GlobalEffects,
        SLIMElasticNetRecommender,
        UserKNNCFRecommender,
        MatrixFactorization_BPR_Cython,
        IALSRecommender,
        MatrixFactorization_SVDpp_Cython,
        # MatrixFactorization_AsySVD_Cython,
        EASE_R_Recommender,
        ItemKNNCFRecommender,
        P3alphaRecommender,
        SLIM_BPR_Cython,
        RP3betaRecommender,
        PureSVDRecommender,
        NMFRecommender,
        UserKNNCBFRecommender,
        ItemKNNCBFRecommender,
        UserKNN_CFCBF_Hybrid_Recommender,
        ItemKNN_CFCBF_Hybrid_Recommender,
        LightFMCFRecommender,
        LightFMUserHybridRecommender,
        LightFMItemHybridRecommender,
        NegHOSLIMRecommender,
        NegHOSLIMElasticNetRecommender,
        # MultVAERecommender,
        ]


    metric_to_optimize = 'NDCG'
    cutoff_to_optimize = 10
    cutoff_list = [5, 10, 20, 30, 40, 50, 100]
    max_total_time = 14*24*60*60  # 14 days
    n_cases = 50
    n_processes = 5
    # split_type = "random_holdout_80_10_10"
    # split_type = "leave_1_out"

    for split_type in ["random_holdout_80_10_10", "leave_1_out"]:

        functions_for_parallel_dataset.read_data_split_and_search(dataset_list,
                                                    recommender_class_list,
                                                    KNN_similarity_to_report_list = KNN_similarity_to_report_list,
                                                    flag_baselines_tune = input_flags.baseline_tune,
                                                    metric_to_optimize = metric_to_optimize,
                                                    cutoff_to_optimize = cutoff_to_optimize,
                                                    cutoff_list = cutoff_list,
                                                    n_cases = n_cases,
                                                    max_total_time = max_total_time,
                                                    n_processes = n_processes,
                                                    resume_from_saved = True,
                                                    split_type = split_type,
                                                    )

        # If the search was done with parallel datasets then this sequential step will
        # only print the result tables
        for dataset_class in dataset_list:
            functions_for_parallel_model.read_data_split_and_search(dataset_class,
                                       recommender_class_list,
                                       KNN_similarity_to_report_list = KNN_similarity_to_report_list,
                                       flag_baselines_tune = False,
                                       flag_print_results = input_flags.print_results,
                                       metric_to_optimize = metric_to_optimize,
                                       cutoff_to_optimize = cutoff_to_optimize,
                                       cutoff_list = cutoff_list,
                                       n_cases = n_cases,
                                       max_total_time = max_total_time,
                                       n_processes = n_processes,
                                       resume_from_saved = True,
                                       split_type = split_type,
                                       )
