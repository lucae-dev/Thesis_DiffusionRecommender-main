from Evaluation.Evaluator import EvaluatorHoldout
import numpy as np
from Data_manager import *
from skopt.space import Real, Integer, Categorical

from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs


from Diffusion.DiffusionRecommender import *
from Data_manager.DataSplitter_Holdout import DataSplitter_Holdout

# https://deci.ai/blog/tricks-training-neural-networks/
# https://www.reddit.com/r/MLQuestions/comments/l3w47e/why_batchnorm_needs_to_set_biasfalse_in_pytorch/

if __name__ == '__main__':

    dataset_list = [
        # 10^4
        FilmTrustReader,
        # FrappeReader,
        # LastFMHetrec2011Reader,

        # 10^5
        # Movielens100KReader,
        # MovielensHetrec2011Reader,
        # CiteULike_aReader,
        # TafengReader,
        # CiteULike_tReader,
        # BrightkiteReader,
        # DeliciousHetrec2011Reader,
        # AmazonMusicalInstrumentsReader,
        # CiaoReader,
        # EpinionsReader,
        # AmazonMusicReader,

        # 10^6
        # JesterJokesReader,
        # Movielens1MReader,
        # PinterestReader,
        # TVAudienceReader,
        # ContentWiseImpressionsReader,
        # YelpReader,
        # AmazonMoviesTVReader,
        # AmazonAutomotiveReader,
        # BookCrossingReader,
        # AmazonElectronicsReader,
        # GowallaReader,

        # 10^7
        # Movielens10MReader,
        # Movielens20MReader,
        # TheMoviesDatasetReader,
        # Movielens25MReader,
        # AmazonBooksReader,

        # 10^8
        # NetflixPrizeReader,

    ]


    for dataset_reader in dataset_list:

        ################################################################
        ####    Read Data split
        ################################################################

        dataset_reader = dataset_reader()

        result_folder_path = "result_experiments/diffusion_recommender/{}/".format(dataset_reader._get_dataset_name())

        # forbid_new_split ensures that a new split is NOT generated if not found.
        dataSplitter = DataSplitter_Holdout(dataset_reader, user_wise = False, split_interaction_quota_list=[80, 10, 10], forbid_new_split=True)
        dataSplitter.load_data(save_folder_path=result_folder_path + "data/")

        URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()

        available_ICM = dataSplitter.get_loaded_ICM_names()
        ICM_train = dataSplitter.get_ICM_from_name(available_ICM[0]) if len(available_ICM) > 0 else None

        available_UCM = dataSplitter.get_loaded_UCM_names()
        UCM_train = dataSplitter.get_UCM_from_name(available_UCM[0]) if len(available_UCM) > 0 else None


        ################################################################
        ####    Set hyperparameter optimization and evaluators
        ################################################################

        metric_to_optimize = 'NDCG'
        cutoff_to_optimize = 10     # Cutoff that will be optimized
        cutoff_list = [5, 10, 20, 30, 40, 50, 100]  # The recommendation quality will be computed for all of these cutoffs and saved
        max_total_time = 14*24*60*60  # Max time to allot for hyperparameter optimizations (in seconds). Currently 14 days
        n_cases = 50            # Number of hyperparameter sets to explore
        n_processes = None      # Parallelize baseline optimization (works only with


        evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list = cutoff_list)
        evaluator_validation_earlystopping = EvaluatorHoldout(URM_validation, cutoff_list = [cutoff_to_optimize])
        evaluator_test = EvaluatorHoldout(URM_test, cutoff_list = cutoff_list)


        earlystopping_keywargs = {"validation_every_n": 5,
                                  "stop_on_validation": True,
                                  "evaluator_object": evaluator_validation_earlystopping,
                                  "lower_validations_allowed": 5,
                                  "validation_metric": metric_to_optimize,
                                  "epochs_min": 200,
                                  }



        ################################################################
        ####    Test one instance to see if everything works
        ################################################################

        # Choose which recommender class to use
        recommender_class = DiffusionSFactorizedRecommender_OptimizerMask


        recommender_instance = recommender_class(URM_train, use_gpu=True)
        recommender_instance.fit(encoding_size = 150,
                                 max_parameters = np.inf,
                                 noise_timesteps = 100,
                                 epochs=10,
                                 batch_size=8,
                                 encoder_architecture=None,
                                 l2_reg=1e-4,
                                 sgd_mode='adam',
                                 learning_rate=1e-2,
                                 inference_timesteps=100,
                                 show_progress_bar=False,
                                 start_beta=0.0001,
                                 end_beta=0.10,
                                 activation_function="ReLU",
                                 use_batch_norm=False,
                                 use_dropout=False,
                                 dropout_p=0.3,
                                 objective="pred_noise",
                                 **earlystopping_keywargs,
                                 )


        result_df, result_string = evaluator_validation.evaluateRecommender(recommender_instance)

        print(result_string)


        ################################################################
        ####    Set hyperparameters space for the optimization
        ################################################################


        n_items = URM_train.shape[1]

        hyperparameters_range_dictionary_diffusion = {
            "epochs": Categorical([1500]),
            "learning_rate": Real(low=1e-6, high=1e-1, prior="log-uniform"),
            "l2_reg": Real(low=1e-6, high=1e-1, prior="log-uniform"),
            "batch_size": Categorical([2, 4, 8, 16, 32, 64, 128]),

            "start_beta": Real(low=1e-4, high=1e-3, prior="log-uniform"),
            "end_beta": Real(low=1e-3, high=1e-0, prior="log-uniform"),
            "noise_timesteps": Real(low=2, high=1e+3, prior="log-uniform"),
            "inference_timesteps": Categorical([1]),

            "objective": Categorical(["pred_x0"]),# Categorical(["pred_noise", "pred_x0"]), # Pred noise does not work because I never completed debugging
            "sgd_mode": Categorical(["sgd", "adagrad", "adam", "rmsprop"]),
            "use_batch_norm": Categorical([False]),
            "use_dropout": Categorical([False]),
            "dropout_p": Real(low=0.1, high=0.9, prior="log-uniform"),

            # Constrain the model to a maximum number of parameters so that its size does not exceed 1.45 GB
            # Estimate size by considering each parameter uses float32
            "max_parameters": Categorical([3.45*1e9*8/32]),
        }

        for recommender_class in [
                                  DiffusionSFactorizedRecommender_OptimizerMask,
                                  DiffusionAsySVDRecommender_OptimizerMask,
                                  DiffusionItemSVDRecommender_OptimizerMask,
                                  DiffusionSDenseRecommender_OptimizerMask,
                                  DiffusionResidualAutoencoderRecommender_OptimizerMask,
                                  DiffusionSNoDiagonalDenseRecommender_OptimizerMask,
                                  SNoDiagonalDenseAndAutoencoderModel_OptimizerMask,
                                  ]:

            if recommender_class in [DiffusionSFactorizedRecommender_OptimizerMask,
                                      DiffusionAsySVDRecommender_OptimizerMask,
                                      DiffusionItemSVDRecommender_OptimizerMask]:

                architecture_range_dictionary = {
                    "encoding_size": Integer(1, min(2048, n_items-1)),
                }

            elif recommender_class in [DiffusionAutoencoderRecommender_OptimizerMask,
                                       DiffusionResidualAutoencoderRecommender_OptimizerMask,
                                       SNoDiagonalDenseAndAutoencoderModel_OptimizerMask,]:

                architecture_range_dictionary = {
                    "encoding_size": Integer(1, min(512, n_items-1)),
                    "next_layer_size_multiplier": Integer(2, 5),
                    "max_n_hidden_layers": Integer(3, 7),
                    "activation_function": Categorical(["ReLU", "LeakyReLU", "Sigmoid", "GELU", "Tanh"]),
                    }

            else:
                architecture_range_dictionary = {}


            hyperparameters_range_dictionary = {**hyperparameters_range_dictionary_diffusion, **architecture_range_dictionary}



            ################################################################
            ####    Set objects needed to create an instance of the recommender
            ################################################################

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {"use_gpu": True, "verbose": False},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {},
                EARLYSTOPPING_KEYWORD_ARGS = earlystopping_keywargs,
            )

            # URM to be used for the final evaluation using test data
            URM_train_last_test = URM_train + URM_validation

            # Objects needed to create the final instance of the recommender to be evaluated on the test data
            # It is identical to the one used when evaluating on the validation except for the URM, which in this
            # Case contains all data (train + validation)
            if URM_train_last_test is not None:
                recommender_input_args_last_test = recommender_input_args.copy()
                recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test
            else:
                recommender_input_args_last_test = None



            ################################################################
            ####    Create the object that will do the hyperparameter optimization and run it
            ################################################################

            hyperparameterSearch = SearchBayesianSkopt(recommender_class, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)

            ## Final step, after the hyperparameter range has been defined for each type of algorithm
            hyperparameterSearch.search(recommender_input_args,
                                   hyperparameter_search_space= hyperparameters_range_dictionary,
                                   n_cases = n_cases,
                                   n_random_starts = int(n_cases/3),
                                   resume_from_saved = True,        # If False, optimization will restart from zero every time
                                   save_model = "best",             # Save only the best model (not all 50)
                                   evaluate_on_test = "best",       # When encountering a new good hyperparameter configuration also evaluate on test (the result is saved but not used to avoid information leakage)
                                   max_total_time = max_total_time,
                                   output_folder_path = result_folder_path + "models/",     # Where to save the actual models
                                   output_file_name_root = recommender_class.RECOMMENDER_NAME,  # Prefix for all files that will be saved, tipically it is the recommender name
                                   metric_to_optimize = metric_to_optimize,
                                   cutoff_to_optimize = cutoff_to_optimize,
                                   recommender_input_args_last_test = recommender_input_args_last_test)





