#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/2018

@author: Maurizio Ferrari Dacrema
"""

import os, shutil
import numpy as np
import unittest

from Recommenders.BaseCBFRecommender import BaseItemCBFRecommender, BaseUserCBFRecommender

from Evaluation.Evaluator import EvaluatorHoldout, EvaluatorNegativeItemSample
from Data_manager.Movielens.Movielens1MReader import Movielens1MReader
from Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.Recommender_import_list import *

def write_log_string(log_file, string):
    log_file.write(string)
    log_file.flush()


def _get_instance(recommender_class, URM_train, ICM_all, UCM_all):

    if issubclass(recommender_class, BaseItemCBFRecommender):
        recommender_object = recommender_class(URM_train, ICM_all)
    elif issubclass(recommender_class, BaseUserCBFRecommender):
        recommender_object = recommender_class(URM_train, UCM_all)
    else:
        recommender_object = recommender_class(URM_train)

    return recommender_object









class MyTestSuite(object):
    recommender_class = None

    @classmethod
    def setUpClass(cls):

        dataset_object = Movielens1MReader()

        dataSplitter = DataSplitter_leave_k_out(dataset_object, k_out_value=2)

        dataSplitter.load_data()
        cls.URM_train, cls.URM_validation, cls.URM_test = dataSplitter.get_holdout_split()
        cls.ICM_all = dataSplitter.get_loaded_ICM_dict()["ICM_genres"]
        cls.UCM_all = dataSplitter.get_loaded_UCM_dict()["UCM_all"]

        evaluator = EvaluatorHoldout(cls.URM_test, [5], exclude_seen = True)
        earlystopping_keywargs = {"epochs": 20,
                                  "validation_every_n": 5,
                                  "stop_on_validation": True,
                                  "evaluator_object": evaluator,
                                  "lower_validations_allowed": 5,
                                  "validation_metric": "NDCG",
                                  }

        cls.temp_path = "./result_experiments/__temp_model/{}/".format(cls.recommender_class.RECOMMENDER_NAME)

        if not os.path.isdir(cls.temp_path):
            os.makedirs(cls.temp_path)

        cls.recommender_instance = _get_instance(cls.recommender_class, cls.URM_train, cls.ICM_all, cls.UCM_all)

        if isinstance(cls.recommender_instance, Incremental_Training_Early_Stopping):
            fit_params = earlystopping_keywargs
        else:
            fit_params = {}

        cls.recommender_instance.fit(**fit_params)

        cls.recommender_instance.save_model(cls.temp_path, file_name="temp_model")


    @classmethod
    def tearDownClass(cls):
         shutil.rmtree(cls.temp_path, ignore_errors = True)


    def test_evaluation(self):

        cutoff_list = [5, 10, 25]

        evaluator = EvaluatorHoldout(self.URM_test, cutoff_list, exclude_seen = True)
        results_df_holdout, results_run_string = evaluator.evaluateRecommender(self.recommender_instance)
        self.assertTrue(np.all(results_df_holdout.index == cutoff_list))

        # self.assertTrue(True, "EvaluatorHoldout passed")


        evaluator = EvaluatorNegativeItemSample(self.URM_test, self.URM_train, cutoff_list, exclude_seen = True)
        results_df_negative, _ = evaluator.evaluateRecommender(self.recommender_instance)
        self.assertTrue(np.all(results_df_negative.index == cutoff_list))
        # self.assertTrue(True, "EvaluatorNegativeItemSample passed")

        self.assertTrue(np.all(results_df_holdout.columns == results_df_negative.columns))



    def test_items_to_compute(self):
        """
        Verifies that
        - The returned scores have the right shape, one per existing item
        - The order of items in the items_to_compute parameter does not affect the result

        :return:
        """

        n_users, n_items = self.URM_train.shape
        cutoff = 50

        items_to_compute_not_sorted = np.random.randint(0, n_items, size = 300)
        items_to_compute_sorted = np.sort(items_to_compute_not_sorted)

        for user_id in range(n_users):
            recommendations_all, scores_all = self.recommender_instance.recommend(user_id, cutoff = cutoff, items_to_compute = None, return_scores = True)

            recommendations_sorted, scores_sorted = self.recommender_instance.recommend(user_id, cutoff = cutoff, items_to_compute = items_to_compute_sorted, return_scores = True)
            recommendations_not_sorted, scores_not_sorted = self.recommender_instance.recommend(user_id, cutoff = cutoff, items_to_compute = items_to_compute_not_sorted, return_scores = True)

            self.assertEqual(scores_all.shape, (1, n_items))
            self.assertEqual(scores_sorted.shape, (1, n_items))
            self.assertEqual(scores_not_sorted.shape, (1, n_items))

            self.assertEqual(len(recommendations_all), cutoff)
            self.assertEqual(len(recommendations_sorted), cutoff)
            self.assertEqual(len(recommendations_not_sorted), cutoff)

            # try:
            self.assertTrue(np.equal(recommendations_sorted, recommendations_not_sorted).all())
            self.assertTrue(np.allclose(scores_sorted, scores_not_sorted, atol=1e-5))

            self.assertTrue(np.allclose(scores_sorted[0,items_to_compute_sorted], scores_all[0,items_to_compute_sorted], atol=1e-5))

            scores_sorted[0,items_to_compute_sorted] = -np.inf
            self.assertTrue(np.isinf(scores_sorted).all())
            # except:
            #     # np.where(np.logical_not(scores_sorted == scores_not_sorted))[1]
            #     pass

        self.assertTrue(True, "Items to compute passed")



    def test_user_batch(self):
        """
        Verifies that
        - The user receives the same item scores when those are computed for a single user or for a batch of users

        :return:
        """

        n_users, n_items = self.URM_train.shape
        cutoff = 50

        batch_size = 164 # just a strange number

        for start_user in range(0, n_users, batch_size):

            end_user = min(start_user+batch_size, n_users)

            user_id_batch = np.arange(start_user, end_user, dtype = np.int)
            recommendations_batch, scores_batch = self.recommender_instance.recommend(user_id_batch, cutoff = cutoff, items_to_compute = None, return_scores = True)

            self.assertEqual(scores_batch.shape, (len(user_id_batch), n_items))

            for user_index_in_batch, user_id in enumerate(user_id_batch):
                recommendations_user, scores_user = self.recommender_instance.recommend(user_id, cutoff = cutoff, items_to_compute = None, return_scores = True)

                self.assertTrue(np.equal(recommendations_batch[user_index_in_batch], recommendations_user).all())
                self.assertTrue(np.allclose(scores_batch[user_index_in_batch], scores_user, atol=1e-5))


        self.assertTrue(True, "User batch passed")




    def test_recommend(self):
        """
        Verifies that
        - The returned scores have the right shape, one per existing item

        :return:
        """

        user_batch = 150
        cutoff = 50
        n_users, n_items = self.URM_train.shape

        for user_id_start in range(0, n_items, user_batch):
            user_id_array = np.arange(user_id_start, min(n_users, user_id_start + user_batch))
            recommendations_all, scores_all = self.recommender_instance.recommend(user_id_array, cutoff = cutoff, items_to_compute = None, return_scores = True)

            self.assertEqual(len(recommendations_all), len(user_id_array))

            for id in range(len(recommendations_all)):
                self.assertEqual(len(recommendations_all[id]), cutoff)

            self.assertEqual(scores_all.shape, (len(user_id_array), n_items))



    def test_save_load_and_evaluate(self):

        cutoff_list = [5, 10, 25]

        evaluator = EvaluatorHoldout(self.URM_test, cutoff_list, exclude_seen = True)
        results_df, results_run_string = evaluator.evaluateRecommender(self.recommender_instance)


        recommender_object = _get_instance(self.recommender_class, self.URM_train, self.ICM_all, self.UCM_all)
        recommender_object.load_model(self.temp_path, file_name="temp_model")

        evaluator = EvaluatorHoldout(self.URM_test, cutoff_list, exclude_seen = True)
        result_df_load, results_run_string_2 = evaluator.evaluateRecommender(recommender_object)

        self.assertTrue(results_df.equals(result_df_load), "Save and load + evaluation passed")


    def test_DataIO_compatible(self):

        from Recommenders.DataIO import DataIO
        dataIO = DataIO(self.temp_path)
        data = dataIO.load_data("temp_model.zip")

        self.assertTrue(True, "DataIO compatibility passed")





class MyTestSuite_stochastic(MyTestSuite):

    def test_items_to_compute(self):
        """
        The random recommender only ensures that the "items_to_compute" are associated to a score
        the recommendation list may of course change between calls
        :return:
        """

        cutoff = 50

        items_to_compute_not_sorted = np.random.randint(0, self.URM_train.shape[1], size = 300)
        items_to_compute_sorted = np.sort(items_to_compute_not_sorted)

        for user_id in range(self.URM_train.shape[0]):
            recommendations_sorted, scores_sorted = self.recommender_instance.recommend(user_id, cutoff = cutoff, items_to_compute = items_to_compute_sorted, return_scores = True)

            self.assertEqual(scores_sorted.shape, (1, self.URM_train.shape[1]))

            self.assertEqual(len(recommendations_sorted), cutoff)

            scores_sorted[0,items_to_compute_sorted] = -np.inf
            self.assertTrue(np.isinf(scores_sorted).all())

        self.assertTrue(True, "Items to compute passed")


    def test_save_load_and_evaluate(self):
        """
        The Random recommender only checks that save and load produces a working recommender model
        but does not check that the scores produced are the same
        :return:
        """

        cutoff_list = [5, 10, 25]

        evaluator = EvaluatorHoldout(self.URM_test, cutoff_list, exclude_seen = True)
        _, _ = evaluator.evaluateRecommender(self.recommender_instance)


        recommender_object = _get_instance(self.recommender_class, self.URM_train, self.ICM_all, self.UCM_all)
        recommender_object.load_model(self.temp_path, file_name="temp_model")

        evaluator = EvaluatorHoldout(self.URM_test, cutoff_list, exclude_seen = True)
        _, _ = evaluator.evaluateRecommender(recommender_object)


    def test_user_batch(self):
        pass


class MyTestSuite_Random(MyTestSuite_stochastic, unittest.TestCase):
    recommender_class = Random

class MyTestSuite_TopPop(MyTestSuite, unittest.TestCase):
    recommender_class = TopPop

class MyTestSuite_GlobalEffects(MyTestSuite, unittest.TestCase):
    recommender_class = GlobalEffects

class MyTestSuite_UserKNNCFRecommender(MyTestSuite, unittest.TestCase):
    recommender_class = UserKNNCFRecommender

class MyTestSuite_ItemKNNCFRecommender(MyTestSuite, unittest.TestCase):
    recommender_class = ItemKNNCFRecommender

class MyTestSuite_UserKNNCBFRecommender(MyTestSuite, unittest.TestCase):
    recommender_class = UserKNNCBFRecommender

class MyTestSuite_ItemKNNCBFRecommender(MyTestSuite, unittest.TestCase):
    recommender_class = ItemKNNCBFRecommender

class MyTestSuite_ItemKNN_CFCBF_Hybrid_Recommender(MyTestSuite, unittest.TestCase):
    recommender_class = ItemKNN_CFCBF_Hybrid_Recommender

class MyTestSuite_UserKNN_CFCBF_Hybrid_Recommender(MyTestSuite, unittest.TestCase):
    recommender_class = UserKNN_CFCBF_Hybrid_Recommender

class MyTestSuite_P3alphaRecommender(MyTestSuite, unittest.TestCase):
    recommender_class = P3alphaRecommender

class MyTestSuite_RP3betaRecommender(MyTestSuite, unittest.TestCase):
    recommender_class = RP3betaRecommender

class MyTestSuite_SLIM_BPR_Cython(MyTestSuite, unittest.TestCase):
    recommender_class = SLIM_BPR_Cython

class MyTestSuite_SLIMElasticNetRecommender(MyTestSuite, unittest.TestCase):
    recommender_class = SLIMElasticNetRecommender

class MyTestSuite_MatrixFactorization_BPR_Cython(MyTestSuite, unittest.TestCase):
    recommender_class = MatrixFactorization_BPR_Cython

class MyTestSuite_MatrixFactorization_FunkSVD_Cython(MyTestSuite, unittest.TestCase):
    recommender_class = MatrixFactorization_FunkSVD_Cython

class MyTestSuite_MatrixFactorization_AsySVD_Cython(MyTestSuite, unittest.TestCase):
    recommender_class = MatrixFactorization_AsySVD_Cython

class MyTestSuite_PureSVDRecommender(MyTestSuite, unittest.TestCase):
    recommender_class = PureSVDRecommender

class MyTestSuite_NMFRecommender(MyTestSuite, unittest.TestCase):
    recommender_class = NMFRecommender

class MyTestSuite_IALSRecommender(MyTestSuite, unittest.TestCase):
    recommender_class = IALSRecommender

class MyTestSuite_EASE_R_Recommender(MyTestSuite, unittest.TestCase):
    recommender_class = EASE_R_Recommender

class MyTestSuite_LightFMCFRecommender(MyTestSuite, unittest.TestCase):
    recommender_class = LightFMCFRecommender

class MyTestSuite_LightFMUserHybridRecommender(MyTestSuite, unittest.TestCase):
    recommender_class = LightFMUserHybridRecommender

class MyTestSuite_LightFMItemHybridRecommender(MyTestSuite, unittest.TestCase):
    recommender_class = LightFMItemHybridRecommender

class MyTestSuite_MultVAERecommender(MyTestSuite, unittest.TestCase):
    recommender_class = MultVAERecommender

class MyTestSuite_NegHOSLIMRecommender(MyTestSuite, unittest.TestCase):
    recommender_class = NegHOSLIMRecommender

from Recommenders.SLIM.NegHOSLIM import NegHOSLIMRecommender_dense

class MyTestSuite_NegHOSLIMRecommender_dense(MyTestSuite, unittest.TestCase):
    recommender_class = NegHOSLIMRecommender_dense

class MyTestSuite_NegHOSLIMElasticNetRecommender(MyTestSuite, unittest.TestCase):
    recommender_class = NegHOSLIMElasticNetRecommender

# class MyTestSuite_NegHOSLIMLSQR(MyTestSuite, unittest.TestCase):
#     recommender_class = NegHOSLIMLSQR

if __name__ == '__main__':

    unittest.main()
