#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/11/2021

@author: Maurizio Ferrari Dacrema

Porting of:
@inbook{10.1145/3460231.3474273,
    author = {Steck, Harald and Liang, Dawen},
    title = {Negative Interactions for Improved Collaborative Filtering: Don’t Go Deeper, Go Higher},
    year = {2021},
    isbn = {9781450384582},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3460231.3474273},
    abstract = { The recommendation-accuracy of collaborative filtering approaches is typically improved
    when taking into account higher-order interactions [5, 6, 9, 10, 11, 16, 18, 24, 25,
    28, 31, 34, 36, 41, 42, 44]. While deep nonlinear models are theoretically able to
    learn higher-order interactions, their capabilities were, however, found to be quite
    limited in practice [5]. Moreover, the use of low-dimensional embeddings in deep networks
    may severely limit their expressiveness [8]. This motivated us in this paper to explore
    a simple extension of linear full-rank models that allow for higher-order interactions
    as additional explicit input-features. Interestingly, we observed that this model-class
    obtained by far the best ranking accuracies on the largest data set in our experiments,
    while it was still competitive with various state-of-the-art deep-learning models
    on the smaller data sets. Moreover, our approach can also be interpreted as a simple
    yet effective improvement of the (linear) HOSLIM [11] model: by simply removing the
    constraint that the learned higher-order interactions have to be non-negative, we
    observed that the accuracy-gains due to higher-order interactions more than doubled
    in our experiments. The reason for this large improvement was that large positive
    higher-order interactions (as used in HOSLIM [11]) are relatively infrequent compared
    to the number of large negative higher-order interactions in the three well-known
    data-sets used in our experiments. We further characterize the circumstances where
    the higher-order interactions provide the most significant improvements.},
    booktitle = {Fifteenth ACM Conference on Recommender Systems},
    pages = {34–43},
    numpages = {10}
}

"""


from sklearn.linear_model import ElasticNet
import time, sys
import numpy as np
import scipy.sparse as sps
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.BaseRecommender import BaseRecommender
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit
from Recommenders.DataIO import DataIO
from copy import deepcopy
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from Recommenders.Similarity.Compute_Similarity_Python import Incremental_Similarity_Builder
from Recommenders.Recommender_utils import check_matrix


#
# ### functions to create the feature-pairs
# def create_list_feature_pairs(XtX, threshold):
#     AA = sps.triu(abs(XtX))
#     AA.setdiag(0.0)
#     AA = AA>threshold
#     AA.eliminate_zeros()
#     ii_pairs = AA.nonzero()
#     return ii_pairs
#
# def create_matrix_Z(ii_pairs, X):
#     MM = sps.lil_matrix((len(ii_pairs[0]), X.shape[1]),    dtype=np.float)
#     MM[np.arange(MM.shape[0]) , ii_pairs[0]   ]=1.0
#     MM[np.arange(MM.shape[0]) , ii_pairs[1]   ]=1.0
#
#     CCmask = 1.0-MM.todense()    # see Eq. 8 in the paper
#
#     MM=sps.csc_matrix(MM.T)
#     Z=  X * MM
#     Z= (Z == 2.0 )
#     Z=Z*1.0
#     return [ Z, CCmask]

### functions to create the feature-pairs
def create_list_feature_pairs_threshold(XtX, threshold):
    AA= np.triu(np.abs(XtX))
    AA[ np.diag_indices(AA.shape[0]) ]=0.0
    ii_pairs = np.where((AA>threshold)==True)
    return ii_pairs

def create_list_feature_pairs_topn(XtX, topn):
    AA= np.triu(np.abs(XtX))
    AA[ np.diag_indices(AA.shape[0]) ]=0.0

    rows, cols = AA.nonzero()
    topn_index = np.argpartition(-AA[rows, cols], topn)[:topn]
    ii_pairs = rows[topn_index], cols[topn_index]

    return ii_pairs



def create_matrix_Z(ii_pairs, X):
    MM = np.zeros( (len(ii_pairs[0]), X.shape[1]),    dtype=np.float)
    MM[np.arange(MM.shape[0]) , ii_pairs[0]   ]=1.0
    MM[np.arange(MM.shape[0]) , ii_pairs[1]   ]=1.0
    CCmask = 1.0-MM    # see Eq. 8 in the paper
    MM=sps.csc_matrix(MM.T)
    Z=  X * MM
    Z= (Z == 2.0 )
    Z=Z*1.0
    return [ Z, CCmask]





def create_list_feature_pairs_threshold_sps(XtX, threshold):
    XtX_diag = XtX.diagonal()
    XtX.setdiag(0.0)
    AA = sps.triu(abs(XtX))
    XtX.setdiag(XtX_diag)

    boolean_index = AA.data>threshold
    ii_pairs = AA.row[boolean_index], AA.col[boolean_index]
    return ii_pairs


def create_list_feature_pairs_topn_sps(XtX, topn):
    XtX_diag = XtX.diagonal()
    XtX.setdiag(0.0)
    AA = sps.triu(abs(XtX))
    XtX.setdiag(XtX_diag)

    topn_index = np.argpartition(-AA.data, topn)[:topn]
    ii_pairs = AA.row[topn_index], AA.col[topn_index]

    return ii_pairs


def create_matrix_Z_sps(ii_pairs, X):
    # MM   = |n_feature_pairs|x|n_items|
    # MM = np.zeros( (len(ii_pairs[0]), X.shape[1]),    dtype=np.float)
    MM = sps.lil_matrix((len(ii_pairs[0]), X.shape[1]),    dtype=np.float)
    MM[np.arange(MM.shape[0]), ii_pairs[0] ]=1.0
    MM[np.arange(MM.shape[0]), ii_pairs[1] ]=1.0

    MM = sps.csc_matrix(MM.T)
    Z =  X * MM
    Z = (Z == 2.0 )
    Z = Z*1.0
    # Z         = |n_users|x|n_feature_pairs|
    # MM        = |n_items|x|n_feature_pairs|
    # CCmask    = |n_items|x|n_feature_pairs|
    CCmask = 1.0-MM.T.toarray()    # see Eq. 8 in the paper
    return Z, MM.T, CCmask





class NegHOSLIMRecommender_dense(BaseRecommender, Incremental_Training_Early_Stopping):
    """
    """

    RECOMMENDER_NAME = "NegHOSLIMRecommender_dense"

    def __init__(self, URM_train, verbose = True):
        super(NegHOSLIMRecommender_dense, self).__init__(URM_train, verbose = verbose)



    def _compute_item_score(self, user_id_array, items_to_compute = None):

        Xtest = self.URM_train[user_id_array,:]
        Ztest = self.Z[user_id_array,:]

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.n_items), dtype=np.float32)*np.inf
            BB_items_to_compute = self.BB[items_to_compute,:][:,items_to_compute]
            item_scores[:, items_to_compute] = Xtest.dot(self.BB[:,items_to_compute]) +  Ztest.dot(self.CC[:, items_to_compute])

        else:
            item_scores = (Xtest).dot(self.BB) + Ztest.dot(self.CC)

        return item_scores


    def _create_feature_pairs(self):

        self.X = self.URM_train

        start_time = time.time()
        self._print("Creating Feature Pairs...")

        self.XtX=np.array( ( self.X.transpose() * self.X).todense())
        self.XtXdiag=deepcopy(np.diag(self.XtX))

        ### create the list of feature-pairs and the higher-order matrix Z
        self.XtX[ np.diag_indices(self.XtX.shape[0]) ] = self.XtXdiag #if code is re-run, ensure that the diagonal is correct

        if self.feature_pairs_threshold is not None:
            ii_feature_pairs = create_list_feature_pairs_threshold(self.XtX, self.feature_pairs_threshold)
        else:
            ii_feature_pairs = create_list_feature_pairs_topn(self.XtX, self.feature_pairs_n)

        # print("number of feature-pairs: {}".format(len(ii_feature_pairs[0])))
        self.Z, self.CCmask = create_matrix_Z(ii_feature_pairs, self.X)
        # Z_test_data_tr , _ = create_matrix_Z(ii_feature_pairs, test_data_tr)

        new_time_value, new_time_unit = seconds_to_biggest_unit(time.time()-start_time)
        self._print("Creating Feature Pairs... done in {:.2f} {}. Number of Feature-Pairs: {}".format( new_time_value, new_time_unit, len(ii_feature_pairs[0])))



    def fit(self, epochs=300, feature_pairs_threshold = None, feature_pairs_n = 100, lambdaBB = 5e2, lambdaCC = 5e3, rho = 1e5,
            **earlystopping_kwargs):

        self.rho = rho

        assert np.logical_xor(feature_pairs_threshold is not None, feature_pairs_n is not None), "feature_pairs_threshold and feature_pairs_n cannot be used at the same time"

        self.feature_pairs_threshold = feature_pairs_threshold
        self.feature_pairs_n = feature_pairs_n

        ####### Data structures summary
        # X     = |n_users|x|n_items|     training data
        # XtX   = |n_items|x|n_items|
        # Z     = |n_users|x|n_feature_pairs|
        # CCmask = |n_feature_pairs|x|n_items|
        # ZtX    = |n_feature_pairs|x|n_items|
        # PP    = |n_items|x|n_items|
        # QQ    = |n_feature_pairs|x|n_feature_pairs|
        # CC    = |n_feature_pairs|x|n_items|
        # DD    = |n_feature_pairs|x|n_items|
        # UU    = |n_feature_pairs|x|n_items|
        # BB    = |n_items|x|n_items|



        self._create_feature_pairs()

        ### create the higher-order matrices
        start_time = time.time()
        self._print("Creating Higher-Order Matrices...")
        ZtZ=np.array(  (self.Z.transpose() * self.Z).todense())
        self.ZtX=np.array( (self.Z.transpose() * self.X).todense())
        ZtZdiag=deepcopy(np.diag(ZtZ))
        self._print("Creating Higher-Order Matrices... done in {:.2f} {}.".format(*seconds_to_biggest_unit(time.time()-start_time)))

        # precompute for BB
        start_time = time.time()
        self._print("Precomputing BB and CC...")
        self.ii_diag=np.diag_indices(self.XtX.shape[0])
        self.XtX[self.ii_diag] = self.XtXdiag+lambdaBB
        self.PP=np.linalg.inv(self.XtX)

        # precompute for CC
        self.ii_diag_ZZ=np.diag_indices(ZtZ.shape[0])
        ZtZ[self.ii_diag_ZZ] = ZtZdiag+lambdaCC+rho
        self.QQ=np.linalg.inv(ZtZ)

        self._print("Precomputing BB and CC... done in {:.2f} {}.".format(*seconds_to_biggest_unit(time.time()-start_time)))

        # initialize
        self.CC = np.zeros( (ZtZ.shape[0], self.XtX.shape[0]),dtype=np.float )
        self.DD = np.zeros( (ZtZ.shape[0], self.XtX.shape[0]),dtype=np.float )
        self.UU = np.zeros( (ZtZ.shape[0], self.XtX.shape[0]),dtype=np.float ) # is Gamma in paper
        self.BB = np.zeros( (self.URM_train.shape[0], self.URM_train.shape[0]),dtype=np.float )

        ########################### Earlystopping

        self._prepare_model_for_validation()
        self._update_best_model()

        self._train_with_early_stopping(epochs,
                                        algorithm_name = self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        self.BB = self.BB_best
        self.CC = self.CC_best


    def _prepare_model_for_validation(self):
        pass


    def _update_best_model(self):
        self.BB_best = self.BB.copy()
        self.CC_best = self.CC.copy()


    def _run_epoch(self, num_epoch):
        #
        # print("epoch {}".format(iter))

        # learn BB
        self.XtX[self.ii_diag] = self.XtXdiag
        self.BB = self.PP.dot(self.XtX-self.ZtX.T.dot(self.CC))
        gamma = np.diag(self.BB) / np.diag(self.PP)
        self.BB -= self.PP *gamma

        # learn CC
        self.CC= self.QQ.dot(self.ZtX-self.ZtX.dot(self.BB) +self.rho *(self.DD-self.UU))

        # learn DD
        self.DD=  self.CC  * self.CCmask
        #DD= np.maximum(0.0, DD) # if you want to enforce non-negative parameters

        # learn UU (is Gamma in paper)
        self.UU+= self.CC-self.DD



    def save_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {"BB": self.BB,
                             "CC": self.CC,
                             "feature_pairs_threshold": self.feature_pairs_threshold,
                             "feature_pairs_n": self.feature_pairs_n,
                            }

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)

        self._print("Saving complete")


    def load_model(self, folder_path, file_name = None):
        super(NegHOSLIMRecommender_dense, self).load_model(folder_path, file_name = file_name)

        self._create_feature_pairs()





class NegHOSLIMRecommender(BaseRecommender, Incremental_Training_Early_Stopping):
    """
    """

    RECOMMENDER_NAME = "NegHOSLIMRecommender"

    def __init__(self, URM_train, verbose = True):
        super(NegHOSLIMRecommender, self).__init__(URM_train, verbose = verbose)



    def _compute_item_score(self, user_id_array, items_to_compute = None):

        Xtest = self.URM_train[user_id_array,:]
        Ztest = self.Z[user_id_array,:]

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.n_items), dtype=np.float32)*np.inf
            item_scores[:, items_to_compute] = Xtest.dot(self.BB[:,items_to_compute]) +  Ztest.dot(self.CC[:, items_to_compute])

        else:
            item_scores = np.array(Xtest.dot(self.BB) + Ztest.dot(self.CC))

        return item_scores


    def _create_feature_pairs(self):

        self.X = self.URM_train.copy()

        start_time = time.time()
        self._print("Creating Feature Pairs...")

        self.XtX = self.URM_train.T.dot(self.URM_train)
        self.XtXdiag = np.array(self.URM_train.power(2).sum(axis=0)).flatten()
        self.XtX.setdiag(self.XtXdiag)

        ### create the list of feature-pairs and the higher-order matrix Z
        if self.feature_pairs_threshold is not None:
            ii_feature_pairs = create_list_feature_pairs_threshold_sps(self.XtX, self.feature_pairs_threshold)
        else:
            ii_feature_pairs = create_list_feature_pairs_topn_sps(self.XtX, self.feature_pairs_n)

        self.n_feature_pairs = len(ii_feature_pairs[0])
        self.Z, self.MM, self.CCmask = create_matrix_Z_sps(ii_feature_pairs, self.X)

        new_time_value, new_time_unit = seconds_to_biggest_unit(time.time()-start_time)
        self._print("Creating Feature Pairs... done in {:.2f} {}. Number of Feature-Pairs: {}".format( new_time_value, new_time_unit, len(ii_feature_pairs[0])))



    def fit(self, epochs=300, feature_pairs_threshold = None, feature_pairs_n = 100, lambdaBB = 5e2, lambdaCC = 5e3, rho = 1e5,
            **earlystopping_kwargs):
        """

        :param epochs:
        :param feature_pairs_threshold:
        :param feature_pairs_n:
        :param lambdaBB:
        :param lambdaCC:
        :param rho:
        :param earlystopping_kwargs:
        :return:

        ####### Data structures summary
        # X     = |n_users|x|n_items|               sparse    Training data

        # XtX   = |n_items|x|n_items|               sparse
        # PP    = |n_items|x|n_items|               dense
        # BB    = |n_items|x|n_items|               dense

        # Z     = |n_users|x|n_feature_pairs|       sparse

        # CCmask = |n_feature_pairs|x|n_items|              dense
        # ZtX    = |n_feature_pairs|x|n_items|              sparse
        # CC     = |n_feature_pairs|x|n_items|              dense
        # DD     = |n_feature_pairs|x|n_items|              dense
        # UU     = |n_feature_pairs|x|n_items|              dense
        # MM     = |n_feature_pairs|x|n_items|              sparse
        # QQ     = |n_feature_pairs|x|n_feature_pairs|      dense

        """

        assert np.logical_xor(feature_pairs_threshold is not None, feature_pairs_n is not None), "feature_pairs_threshold and feature_pairs_n cannot be used at the same time"

        self.rho = rho
        self.feature_pairs_threshold = feature_pairs_threshold
        self.feature_pairs_n = feature_pairs_n

        self._create_feature_pairs()

        start_time = time.time()

        self._print("Creating Higher-Order Matrices...")
        ZtZ = self.Z.T.dot(self.Z)
        ZtZdiag = deepcopy(ZtZ.diagonal())
        self.ZtX = self.Z.T.dot(self.X).tocsc()

        self._print("Creating Higher-Order Matrices... done in {:.2f} {}.".format(*seconds_to_biggest_unit(time.time()-start_time)))

        start_time = time.time()

        self._print("Precomputing BB and CC...")
        self.XtX.setdiag(self.XtXdiag + lambdaBB)
        self.PP = np.linalg.inv(self.XtX.toarray())
        self.XtX.setdiag(self.XtXdiag)

        ZtZ.setdiag(ZtZdiag + lambdaCC + rho)
        self.QQ = np.linalg.inv(ZtZ.toarray())

        self._print("Precomputing BB and CC... done in {:.2f} {}.".format(*seconds_to_biggest_unit(time.time()-start_time)))

        # initialize
        self.CC = np.zeros((self.n_feature_pairs, self.n_items), dtype=np.float )
        self.DD = np.zeros((self.n_feature_pairs, self.n_items), dtype=np.float )
        self.UU = np.zeros((self.n_feature_pairs, self.n_items), dtype=np.float ) # is Gamma in paper
        self.BB = np.zeros((self.n_items, self.n_items), dtype=np.float )

        ########################### Earlystopping

        self._prepare_model_for_validation()
        self._update_best_model()

        self._train_with_early_stopping(epochs,
                                        algorithm_name = self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        self.BB = self.BB_best
        self.CC = self.CC_best


    def _prepare_model_for_validation(self):
        pass


    def _update_best_model(self):
        self.BB_best = self.BB.copy()
        self.CC_best = self.CC.copy()


    def _run_epoch(self, num_epoch):

        # learn BB
        self.BB = self.PP.dot(self.XtX - self.ZtX.T.dot(self.CC))
        gamma = np.diag(self.BB) / np.diag(self.PP)
        self.BB -= self.PP *gamma

        # learn CC
        self.CC = np.array(self.QQ.dot(self.ZtX - self.ZtX.dot(self.BB) + self.rho * (self.DD - self.UU)))

        # learn DD
        self.DD = np.multiply(self.CC, self.CCmask)
        # DD = np.maximum(0.0, DD) # if you want to enforce non-negative parameters

        # learn UU (is Gamma in paper)
        # self.UU += self.CC-DD
        # simplify UU+= CC-DD = CC - CC*CCmask = CC*(1-CCmask) = CC*(1-1+MM.T) == CC*MM.T        (precompute transpose)
        self.UU += self.MM.multiply(self.CC).toarray()



    def save_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {"BB": self.BB,
                             "CC": self.CC,
                             "feature_pairs_threshold": self.feature_pairs_threshold,
                             "feature_pairs_n": self.feature_pairs_n,
                            }

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)

        self._print("Saving complete")


    def load_model(self, folder_path, file_name = None):
        super(NegHOSLIMRecommender, self).load_model(folder_path, file_name = file_name)

        self._create_feature_pairs()















class NegHOSLIMElasticNetRecommender(BaseRecommender):
    """
    This method combines the idea of NegHO SLIM with a simple ElasticNet solver.
    The originally proposed method called NegHOSLIM can be solved more efficiently but has a high memory requirement,
    this variant retains the higher order relations but learns the weights with a simple ElasticNet solver.

    @inbook{10.1145/3460231.3474273,
        author = {Steck, Harald and Liang, Dawen},
        title = {Negative Interactions for Improved Collaborative Filtering: Don’t Go Deeper, Go Higher},
        year = {2021},
        isbn = {9781450384582},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        url = {https://doi.org/10.1145/3460231.3474273},
        booktitle = {Fifteenth ACM Conference on Recommender Systems},
        pages = {34–43},
        numpages = {10}
    }

    """

    RECOMMENDER_NAME = "NegHOSLIMElasticNetRecommender"

    def __init__(self, URM_train, verbose = True):
        super(NegHOSLIMElasticNetRecommender, self).__init__(URM_train, verbose = verbose)


    def _compute_item_score(self, user_id_array, items_to_compute = None):

        Xtest = self.URM_train[user_id_array,:]
        Ztest = self.Z[user_id_array,:]

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.n_items), dtype=np.float32)*np.inf
            item_scores[:, items_to_compute] = (Xtest.dot(self.W_I_sparse[:,items_to_compute]) + Ztest.dot(self.W_Z_sparse[:, items_to_compute])).toarray()

        else:
            item_scores = (Xtest.dot(self.W_I_sparse) + Ztest.dot(self.W_Z_sparse)).toarray()

        return item_scores


    def _create_feature_pairs(self):

        start_time = time.time()
        self._print("Creating Feature Pairs...")

        XtX = self.URM_train.T.dot(self.URM_train)

        ### create the list of feature-pairs and the higher-order matrix Z
        if self.feature_pairs_threshold is not None:
            ii_feature_pairs = create_list_feature_pairs_threshold_sps(XtX, self.feature_pairs_threshold)
        else:
            ii_feature_pairs = create_list_feature_pairs_topn_sps(XtX, self.feature_pairs_n)

        self.n_feature_pairs = len(ii_feature_pairs[0])
        self.Z, _, _ = create_matrix_Z_sps(ii_feature_pairs, self.URM_train)

        new_time_value, new_time_unit = seconds_to_biggest_unit(time.time()-start_time)
        self._print("Creating Feature Pairs... done in {:.2f} {}. Number of Feature-Pairs: {}".format( new_time_value, new_time_unit, len(ii_feature_pairs[0])))


    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, l1_ratio=0.1, alpha = 1.0, positive_only_weights = False, topK = 100, feature_pairs_threshold = None, feature_pairs_n = 1000):

        assert np.logical_xor(feature_pairs_threshold is not None, feature_pairs_n is not None), "feature_pairs_threshold and feature_pairs_n cannot be used at the same time"
        assert l1_ratio>= 0 and l1_ratio<=1, "{}: l1_ratio must be between 0 and 1, provided value was {}".format(self.RECOMMENDER_NAME, l1_ratio)

        self.feature_pairs_threshold = feature_pairs_threshold
        self.feature_pairs_n = feature_pairs_n
        self.l1_ratio = l1_ratio
        self.positive_only_weights = positive_only_weights
        self.topK = topK

        ####### Data structures summary
        # X     = |n_users|x|n_items|               sparse    Training data
        # Z     = |n_users|x|n_feature_pairs|       sparse

        # WIZ   = |n_items + n_feature_pairs|x|n_items + n_feature_pairs|       sparse
        #         The matrix is defined in blocks like this
        #         WIZ=  | W_I   0  |
        #               | W_Z   0  |
        #
        # W_I     = |n_items|x|n_items|             The usual item-item similarity
        # W_Z     = |n_feature_pairs|x|n_items|     The weights of the higher order feature_pairs


        self._create_feature_pairs()

        # initialize the ElasticNet model
        self.model = ElasticNet(alpha=alpha,
                                l1_ratio=self.l1_ratio,
                                positive=self.positive_only_weights,
                                fit_intercept=False,
                                copy_X=False,
                                precompute=True,
                                selection='random',
                                max_iter=100,
                                tol=1e-4)

        URM_train_Z = sps.hstack([self.URM_train, self.Z])
        URM_train_Z = check_matrix(URM_train_Z, 'csc', dtype=np.float32)

        similarity_builder = Incremental_Similarity_Builder(self.n_items + self.n_feature_pairs,
                                                            initial_data_block=self.n_items*self.topK,
                                                            dtype = np.float32)

        start_time = time.time()
        start_time_printBatch = start_time

        # fit each item's factors sequentially (not in parallel)
        for current_item in range(self.n_items):

            # get the target column
            y = URM_train_Z[:, current_item].toarray()

            # set the j-th column of X to zero
            start_pos = URM_train_Z.indptr[current_item]
            end_pos = URM_train_Z.indptr[current_item + 1]

            current_item_data_backup = URM_train_Z.data[start_pos: end_pos].copy()
            URM_train_Z.data[start_pos: end_pos] = 0.0

            # fit one ElasticNet model per column
            self.model.fit(URM_train_Z, y)

            # self.model.coef_ contains the coefficient of the ElasticNet model
            # let's keep only the non-zero values
            nonzero_model_coef_index = self.model.sparse_coef_.indices
            nonzero_model_coef_value = self.model.sparse_coef_.data

            # Check if there are more data points than topK, if so, extract the set of K best values
            if len(nonzero_model_coef_value) > self.topK:
                # Partition the data because this operation does not require to fully sort the data
                relevant_items_partition = np.argpartition(-np.abs(nonzero_model_coef_value), self.topK-1, axis=0)[0:self.topK]
                nonzero_model_coef_index = nonzero_model_coef_index[relevant_items_partition]
                nonzero_model_coef_value = nonzero_model_coef_value[relevant_items_partition]

            similarity_builder.add_data_lists(row_list_to_add=nonzero_model_coef_index,
                                              col_list_to_add=np.ones(len(nonzero_model_coef_index), dtype = np.int) * current_item,
                                              data_list_to_add=nonzero_model_coef_value)


            # finally, replace the original values of the j-th column
            URM_train_Z.data[start_pos:end_pos] = current_item_data_backup

            elapsed_time = time.time() - start_time
            new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)


            if time.time() - start_time_printBatch > 300 or current_item == self.n_items-1:
                self._print("Processed {} ({:4.1f}%) in {:.2f} {}. Items per second: {:.2f}".format(
                    current_item+1,
                    100.0*float(current_item+1)/self.n_items,
                    new_time_value,
                    new_time_unit,
                    float(current_item)/elapsed_time))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()

        W_I_Z_sparse = similarity_builder.get_SparseMatrix()

        assert W_I_Z_sparse[:,-self.n_feature_pairs:].nnz == 0, "W_Z_sparse contains data where it should not"

        W_I_Z_sparse = W_I_Z_sparse[:,:self.n_items]

        self.W_I_sparse = W_I_Z_sparse[:self.n_items,:]
        self.W_Z_sparse = W_I_Z_sparse[-self.n_feature_pairs:,:]

        self._print("Item-item weight matrix density {:4.1f} E-4, Higher-order weight matrix density {:4.1f} E-4.".format(
            1e+4*float(self.W_I_sparse.nnz)/self.n_items**2,
            1e+4*float(self.W_Z_sparse.nnz)/(self.n_items*self.n_feature_pairs),
        ))






    def save_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {"W_I_sparse": self.W_I_sparse,
                             "W_Z_sparse": self.W_Z_sparse,
                             "feature_pairs_threshold": self.feature_pairs_threshold,
                             "feature_pairs_n": self.feature_pairs_n,
                            }

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)

        self._print("Saving complete")


    def load_model(self, folder_path, file_name = None):
        super(NegHOSLIMElasticNetRecommender, self).load_model(folder_path, file_name = file_name)

        self._create_feature_pairs()















from scipy.sparse.linalg import lsqr as sps_lsqr


class NegHOSLIMLSQR(NegHOSLIMElasticNetRecommender):
    """
    This method combines the idea of NegHO SLIM with a simple ElasticNet solver.
    The originally proposed method called NegHOSLIM can be solved more efficiently but has a high memory requirement,
    this variant retains the higher order relations but learns the weights with a simple ElasticNet solver.

    @inbook{10.1145/3460231.3474273,
        author = {Steck, Harald and Liang, Dawen},
        title = {Negative Interactions for Improved Collaborative Filtering: Don’t Go Deeper, Go Higher},
        year = {2021},
        isbn = {9781450384582},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        url = {https://doi.org/10.1145/3460231.3474273},
        booktitle = {Fifteenth ACM Conference on Recommender Systems},
        pages = {34–43},
        numpages = {10}
    }

    """

    RECOMMENDER_NAME = "NegHOSLIMLSQR"


    def fit(self, damp = 1.0, topK = 100, feature_pairs_threshold = None, feature_pairs_n = 1000):

        assert np.logical_xor(feature_pairs_threshold is not None, feature_pairs_n is not None), "feature_pairs_threshold and feature_pairs_n cannot be used at the same time"

        self.feature_pairs_threshold = feature_pairs_threshold
        self.feature_pairs_n = feature_pairs_n
        self.topK = topK

        ####### Data structures summary
        # X     = |n_users|x|n_items|               sparse    Training data
        # Z     = |n_users|x|n_feature_pairs|       sparse

        # WIZ   = |n_items + n_feature_pairs|x|n_items + n_feature_pairs|       sparse
        #         The matrix is defined in blocks like this
        #         WIZ=  | W_I   0  |
        #               | W_Z   0  |
        #
        # W_I     = |n_items|x|n_items|             The usual item-item similarity
        # W_Z     = |n_feature_pairs|x|n_items|     The weights of the higher order feature_pairs


        self._create_feature_pairs()

        URM_train_Z = sps.hstack([self.URM_train, self.Z])
        URM_train_Z = check_matrix(URM_train_Z, 'csc', dtype=np.float32)

        similarity_builder = Incremental_Similarity_Builder(self.n_items + self.n_feature_pairs,
                                                            initial_data_block=self.n_items*self.topK,
                                                            dtype = np.float32)

        start_time = time.time()
        start_time_printBatch = start_time

        # fit each item's factors sequentially (not in parallel)
        for current_item in range(self.n_items):

            # get the target column
            y = URM_train_Z[:, current_item]

            if y.nnz != 0:

                y = y.toarray()

                # set the j-th column of X to zero
                start_pos = URM_train_Z.indptr[current_item]
                end_pos = URM_train_Z.indptr[current_item + 1]

                current_item_data_backup = URM_train_Z.data[start_pos: end_pos].copy()
                URM_train_Z.data[start_pos: end_pos] = 0.0

                # fit one ElasticNet model per column
                x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var = sps_lsqr(URM_train_Z, y,
                                                                                            iter_lim = 100,
                                                                                            atol = 1e-4,
                                                                                            damp = damp)

                # x contains the coefficients
                # let's keep only the non-zero values
                nonzero_model_coef_index = x.nonzero()[0]
                nonzero_model_coef_value = x[nonzero_model_coef_index]

                # Check if there are more data points than topK, if so, extract the set of K best values
                if len(nonzero_model_coef_value) > self.topK:
                    # Partition the data because this operation does not require to fully sort the data
                    relevant_items_partition = np.argpartition(-np.abs(nonzero_model_coef_value), self.topK-1, axis=0)[0:self.topK]
                    nonzero_model_coef_index = nonzero_model_coef_index[relevant_items_partition]
                    nonzero_model_coef_value = nonzero_model_coef_value[relevant_items_partition]

                similarity_builder.add_data_lists(row_list_to_add=nonzero_model_coef_index,
                                                  col_list_to_add=np.ones(len(nonzero_model_coef_index), dtype = np.int) * current_item,
                                                  data_list_to_add=nonzero_model_coef_value)


                # finally, replace the original values of the j-th column
                URM_train_Z.data[start_pos:end_pos] = current_item_data_backup


            elapsed_time = time.time() - start_time
            new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)


            if time.time() - start_time_printBatch > 300 or current_item == self.n_items-1:
                self._print("Processed {} ({:4.1f}%) in {:.2f} {}. Items per second: {:.2f}".format(
                    current_item+1,
                    100.0*float(current_item+1)/self.n_items,
                    new_time_value,
                    new_time_unit,
                    float(current_item)/elapsed_time))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()

        W_I_Z_sparse = similarity_builder.get_SparseMatrix()

        assert W_I_Z_sparse[:,-self.n_feature_pairs:].nnz == 0, "W_Z_sparse contains data where it should not"

        W_I_Z_sparse = W_I_Z_sparse[:,:self.n_items]

        self.W_I_sparse = W_I_Z_sparse[:self.n_items,:]
        self.W_Z_sparse = W_I_Z_sparse[-self.n_feature_pairs:,:]

        self._print("Item-item weight matrix density {:4.1f} E-4, Higher-order weight matrix density {:4.1f} E-4.".format(
            1e+4*float(self.W_I_sparse.nnz)/self.n_items**2,
            1e+4*float(self.W_Z_sparse.nnz)/(self.n_items*self.n_feature_pairs),
        ))



