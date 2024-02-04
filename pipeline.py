#list of inference timestep to try
#list of traditional recommenders
#list of ddpm recommeders
#number of items to cutoff in first level



from Evaluation.Evaluator import EvaluatorHoldout
from Evaluation.EvaluatorMultipleCarousels import EvaluatorMultipleCarousels
from Recommenders.NonPersonalizedRecommender import TopPop
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

from Data_manager import *

if __name__ == '__main__':

    from Data_manager.DataSplitter_Holdout import DataSplitter_Holdout

    dataset_reader = Movielens1MReader()

    dataSplitter = DataSplitter_Holdout(dataset_reader, user_wise=False, split_interaction_quota_list=[80, 10, 10])
    dataSplitter.load_data()
    URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()

    cutoff_list = [5, 10, 20, 30, 40, 50, 100]

    # Train the model that will be used as first carousel, in this case a top pop
    recommender_toppop = TopPop(URM_train + URM_validation)
    recommender_toppop.fit()

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
    evaluator_test_long_tail = EvaluatorMultipleCarousels(URM_test, cutoff_list=[20], # EvaluatorMultipleCarousels requires a single cutoff value due to limitations in the implementation
                                                          carousel_recommender_list = [recommender_toppop])




    # Replace this with an instance of the model you want to evaluate
    recommender_to_evaluate = ItemKNNCFRecommender(URM_train + URM_validation)
    recommender_to_evaluate.fit()


    result_df, result_string = evaluator_test.evaluateRecommender(recommender_to_evaluate)
    print("Traditional evaluation\n")
    print(result_string)

    result_df, result_string = evaluator_test_long_tail.evaluateRecommender(recommender_to_evaluate)
    print("Carousel evaluation\n")
    print(result_string)
