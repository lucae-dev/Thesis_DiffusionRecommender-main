#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 28/07/2022

@author: Maurizio Ferrari Dacrema
"""

import time, sys
import numpy as np
import scipy.sparse as sps
from Recommenders.DataIO import DataIO

from Recommenders.Recommender_utils import check_matrix
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit
from sklearn.utils.extmath import randomized_svd

from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.Similarity.Compute_Similarity_Python import Incremental_Similarity_Builder




class GraphFilterCF_W_Recommender(BaseItemSimilarityMatrixRecommender):
    """ GraphFilterCFRecommender

    @inproceedings{DBLP:conf/cikm/ShenWZSZLL21,
          author    = {Yifei Shen and
                       Yongji Wu and
                       Yao Zhang and
                       Caihua Shan and
                       Jun Zhang and
                       Khaled B. Letaief and
                       Dongsheng Li},
          editor    = {Gianluca Demartini and
                       Guido Zuccon and
                       J. Shane Culpepper and
                       Zi Huang and
                       Hanghang Tong},
          title     = {How Powerful is Graph Convolution for Recommendation?},
          booktitle = {{CIKM} '21: The 30th {ACM} International Conference on Information
                       and Knowledge Management, Virtual Event, Queensland, Australia, November
                       1 - 5, 2021},
          pages     = {1619--1629},
          publisher = {{ACM}},
          year      = {2021},
          url       = {https://doi.org/10.1145/3459637.3482264},
          doi       = {10.1145/3459637.3482264},
          timestamp = {Fri, 08 Jul 2022 11:26:57 +0200},
          biburl    = {https://dblp.org/rec/conf/cikm/ShenWZSZLL21.bib},
          bibsource = {dblp computer science bibliography, https://dblp.org}
    }

    Note: The original formulation does not apply topK

    """

    RECOMMENDER_NAME = "GraphFilterCF_W_Recommender"

    def __init__(self, URM_train, verbose = True):
        super(GraphFilterCF_W_Recommender, self).__init__(URM_train, verbose = verbose)


    def fit(self, alpha=1.0, num_factors=50, random_seed = None, topK=100):

        start_time = time.time()

        # The paper defines it as a sum, but it may be due to the implicit data
        D_I = np.sqrt(np.array(self.URM_train.sum(axis = 0))).squeeze()
        D_I_inv = 1/(D_I + 1e-6)
        D_U_inv = 1/np.sqrt(np.array(self.URM_train.sum(axis = 1))).squeeze() + 1e-6

        D_I = sps.diags(D_I)
        D_I_inv = sps.diags(D_I_inv)
        D_U_inv = sps.diags(D_U_inv)

        R_tilde = D_U_inv.dot(self.URM_train).dot(D_I_inv)

        U, Sigma, V = randomized_svd(R_tilde,
                                     n_components = num_factors,
                                     random_state = random_seed)

        if topK is None:
            # W = R.T * R + alpha D_I^-1/2 * V * V.T * D_I^+1/2
            # Inverting the product to use the dot on the sparse D_I. Remind that AB = (B.T * A.T).T
            self.W_sparse = R_tilde.T.dot(R_tilde) + alpha * D_I.T.dot((D_I_inv).dot(V.T.dot(V)).T).T

            self.W_sparse = check_matrix(self.W_sparse, format='npy', dtype=np.float32)
            self._W_sparse_format_checked = True
            self._compute_item_score = self._compute_score_W_dense

            new_time_value, new_time_unit = seconds_to_biggest_unit(time.time()-start_time)
            self._print("Computing item-item similarity... done in {:.2f} {}".format(new_time_value, new_time_unit))

        else:
            self.W_sparse = self._compute_W_sparse_with_topk(R_tilde, V, D_I, D_I_inv, alpha, topK)





    def _compute_score_W_dense(self, user_id_array, items_to_compute = None):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        self._check_format()

        user_profile_array = self.URM_train[user_id_array]

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32)*np.inf
            item_scores_all = user_profile_array.dot(self.W_sparse)#.toarray()
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = user_profile_array.dot(self.W_sparse)#.toarray()

        return item_scores


    def _compute_W_sparse_with_topk(self, R_tilde, V, D_I, D_I_inv, alpha, topK = 100):

        n_factors, n_items = V.shape

        similarity_builder = Incremental_Similarity_Builder(n_items, initial_data_block=n_items*topK, dtype = np.float32)

        block_size = 100

        start_item = 0
        end_item = 0

        start_time = time.time()
        start_time_printBatch = start_time

        D_I = sps.csr_matrix(D_I)
        D_I_inv = sps.csr_matrix(D_I_inv)

        # Compute all similarities for each item using vectorization
        while start_item < n_items:

            end_item = min(n_items, start_item + block_size)

            # this_block_weight = np.dot(ITEM_factors[start_item:end_item, :], ITEM_factors.T)
            VtV = np.dot(V.T, V[:,start_item:end_item])

            # W = R.T * R + alpha D_I^-1/2 * V * V.T * D_I^+1/2
            # Inverting the product to use the dot on the sparse D_I. Remind that AB = (B.T * A.T).T
            this_block_weight = R_tilde.T.dot(R_tilde[:, start_item:end_item]) + \
                                alpha * D_I[start_item:end_item,:][:,start_item:end_item].T.dot(D_I_inv.dot(VtV).T).T

            this_block_weight = np.array(this_block_weight)

            # full = R_tilde.T.dot(R_tilde) + alpha * D_I.T.dot((D_I_inv).dot(V.T.dot(V)).T).T
            # assert np.allclose(np.array(this_block_weight), np.array(full[:, start_item:end_item]), atol=1e-5)

            for col_index_in_block in range(this_block_weight.shape[1]):

                this_column_weights = this_block_weight[:, col_index_in_block]
                item_original_index = start_item + col_index_in_block

                # Select TopK
                relevant_items_partition = np.argpartition(-this_column_weights, topK-1, axis=0)[0:topK]
                this_column_weights = this_column_weights[relevant_items_partition]

                # Incrementally build sparse matrix, do not add zeros
                if np.any(this_column_weights == 0.0):
                    non_zero_mask = this_column_weights != 0.0
                    relevant_items_partition = relevant_items_partition[non_zero_mask]
                    this_column_weights = this_column_weights[non_zero_mask]

                similarity_builder.add_data_lists(row_list_to_add=relevant_items_partition,
                              col_list_to_add=np.ones(len(relevant_items_partition), dtype = np.int) *  item_original_index,
                              data_list_to_add=this_column_weights)


            if time.time() - start_time_printBatch > 300 or end_item == n_items:
                new_time_value, new_time_unit = seconds_to_biggest_unit(time.time() - start_time)

                self._print("Item-item similarity column {} ({:4.1f}%), {:.2f} column/sec. Elapsed time {:.2f} {}".format(
                    end_item,
                    100.0 * float(end_item) / n_items,
                    float(end_item) / (time.time() - start_time),
                    new_time_value, new_time_unit))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()


            start_item += block_size

        return similarity_builder.get_SparseMatrix()


    def load_model(self, folder_path, file_name = None):
        super(GraphFilterCF_W_Recommender, self).load_model(folder_path, file_name = file_name)

        if not sps.issparse(self.W_sparse):
            self._W_sparse_format_checked = True
            self._compute_item_score = self._compute_score_W_dense




class GraphFilterCFRecommender(BaseItemSimilarityMatrixRecommender):
    """ GraphFilterCF_Recommender

    @inproceedings{DBLP:conf/cikm/ShenWZSZLL21,
          author    = {Yifei Shen and
                       Yongji Wu and
                       Yao Zhang and
                       Caihua Shan and
                       Jun Zhang and
                       Khaled B. Letaief and
                       Dongsheng Li},
          editor    = {Gianluca Demartini and
                       Guido Zuccon and
                       J. Shane Culpepper and
                       Zi Huang and
                       Hanghang Tong},
          title     = {How Powerful is Graph Convolution for Recommendation?},
          booktitle = {{CIKM} '21: The 30th {ACM} International Conference on Information
                       and Knowledge Management, Virtual Event, Queensland, Australia, November
                       1 - 5, 2021},
          pages     = {1619--1629},
          publisher = {{ACM}},
          year      = {2021},
          url       = {https://doi.org/10.1145/3459637.3482264},
          doi       = {10.1145/3459637.3482264},
          timestamp = {Fri, 08 Jul 2022 11:26:57 +0200},
          biburl    = {https://dblp.org/rec/conf/cikm/ShenWZSZLL21.bib},
          bibsource = {dblp computer science bibliography, https://dblp.org}
    }

    """

    RECOMMENDER_NAME = "GraphFilterCFRecommender"

    def __init__(self, URM_train, verbose = True):
        super(GraphFilterCFRecommender, self).__init__(URM_train, verbose = verbose)


    def fit(self, alpha=1.0, num_factors=50, random_seed = None):

        start_time = time.time()
        self._print("Computing SVD decomposition of the normalized adjacency matrix...")

        self.alpha = alpha

        # The paper defines it as a sum, but it may be due to the implicit data
        self.D_I = np.sqrt(np.array(self.URM_train.sum(axis = 0))).squeeze()
        self.D_I_inv = 1/(self.D_I + 1e-6)
        self.D_U_inv = 1/np.sqrt(np.array(self.URM_train.sum(axis = 1))).squeeze() + 1e-6

        self.D_I = sps.diags(self.D_I)
        self.D_I_inv = sps.diags(self.D_I_inv)
        self.D_U_inv = sps.diags(self.D_U_inv)

        self.R_tilde = self.D_U_inv.dot(self.URM_train).dot(self.D_I_inv)

        _, _, self.V = randomized_svd(self.R_tilde,
                                     n_components = num_factors,
                                     random_state = random_seed)

        self.D_I = sps.csr_matrix(self.D_I)
        self.D_I_inv = sps.csr_matrix(self.D_I_inv)

        new_time_value, new_time_unit = seconds_to_biggest_unit(time.time()-start_time)
        self._print("Computing SVD decomposition of the normalized adjacency matrix... done in {:.2f} {}".format( new_time_value, new_time_unit))


    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        user_profile_array = self.URM_train[user_id_array]
        item_scores_all = - np.ones((len(user_id_array), self.n_items), dtype=np.float32)*np.inf

        block_size = 100

        # Compute all similarities for each item using vectorization
        for start_item in range(0, self.n_items, block_size):
            end_item = min(self.n_items, start_item + block_size)

            # this_block_weight = np.dot(ITEM_factors[start_item:end_item, :], ITEM_factors.T)
            VtV = np.dot(self.V.T, self.V[:,start_item:end_item])

            # W = R.T * R + alpha D_I^-1/2 * V * V.T * D_I^+1/2
            # Inverting the product to use the dot on the sparse D_I. Remind that AB = (B.T * A.T).T
            this_block_weight = self.R_tilde.T.dot(self.R_tilde[:, start_item:end_item]) + \
                                self.alpha * self.D_I[start_item:end_item,:][:,start_item:end_item].T.dot(self.D_I_inv.dot(VtV).T).T

            this_block_weight = np.array(this_block_weight)
            item_scores_all[:,start_item:end_item] = user_profile_array.dot(this_block_weight)

        # This is a terrible solution but reimplementing the computation only for a specific set of items would require some rewriting
        if items_to_compute is not None:
            item_scores_selected = - np.ones((len(user_id_array), self.n_items), dtype=np.float32)*np.inf
            item_scores_selected[:,items_to_compute] = item_scores_all[:,items_to_compute]

            return item_scores_selected
        else:
            return item_scores_all



    def save_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {"alpha": self.alpha,
                             "D_I": self.D_I,
                             "D_I_inv": self.D_I_inv,
                             "D_U_inv": self.D_U_inv,
                             "V": self.V,
                             }

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)

        self._print("Saving complete")



    def load_model(self, folder_path, file_name = None):
        super(GraphFilterCFRecommender, self).load_model(folder_path, file_name = file_name)

        self.R_tilde = self.D_U_inv.dot(self.URM_train).dot(self.D_I_inv)

