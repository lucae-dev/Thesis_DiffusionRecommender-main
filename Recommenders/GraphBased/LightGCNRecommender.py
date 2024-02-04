#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/12/2022

@author: Maurizio Ferrari Dacrema
"""

import scipy.sparse as sps
import numpy as np

from Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
import torch
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity
from Utils.PyTorch.utils import get_optimizer, loss_BPR, loss_MSE
from Utils.PyTorch.Cython.DataIterator import BPRIterator as BPRIterator_cython, InteractionIterator as InteractionIterator_cython, InteractionAndNegativeIterator as InteractionAndNegativeIterator_cython
from Utils.PyTorch.DataIterator import BPRIterator, InteractionIterator, InteractionAndNegativeIterator


def loss_softplus(model, batch, device):
    user, item_positive, item_negative = batch

    # Compute prediction for each element in batch
    all_items = torch.cat([item_positive, item_negative]).to(device)
    all_users = torch.cat([user, user]).to(device)
    
    all_predictions = model.forward(all_users, all_items)
    x_i, x_j = torch.split(all_predictions, [len(user), len(user)])
    
    # pos_item_embedding, neg_item_embedding = torch.split(_model.E_0[all_items], [len(user), len(user)])
    # user_embedding = _model.E_0[user]
    #
    # reg_loss = (1/2)*(user_embedding.norm(2).pow(2) +
    #                   pos_item_embedding.norm(2).pow(2) +
    #                   neg_item_embedding.norm(2).pow(2))/float(len(user))

    # Compute total loss for batch
    loss = torch.mean(torch.nn.functional.softplus(x_j - x_i))

    return loss # + reg_loss*weight_decay


class _LightGCNModel(torch.nn.Module):

    def __init__(self, URM_train, K, embedding_size, device):
        super(_LightGCNModel, self).__init__()
        
        self.K = K
        self.n_users, self.n_items = URM_train.shape
        
        Zero_u_u_sps = sps.csr_matrix((self.n_users, self.n_users))
        Zero_i_i_sps = sps.csr_matrix((self.n_items, self.n_items))
        A = sps.bmat([[Zero_u_u_sps,     URM_train ],
                      [URM_train.T, Zero_i_i_sps   ]], format="csr")

        D_inv = 1/(np.sqrt(np.array(A.sum(axis = 1)).squeeze()) + 1e-6)
        self.A_tilde = sps.diags(D_inv).dot(A).dot(sps.diags(D_inv)).astype(np.float32)
        
        self.E_0 = torch.nn.Parameter(torch.empty((self.n_users + self.n_items, embedding_size), device = device))
        torch.nn.init.normal_(self.E_0, std=0.1)
        
        self.A_tilde = sps.coo_matrix(self.A_tilde)
        self.A_tilde = torch.sparse_coo_tensor(np.vstack([self.A_tilde.row, self.A_tilde.col]), self.A_tilde.data, self.A_tilde.shape, device = device)
        self.A_tilde = self.A_tilde.coalesce()
        
        self.E_final_detached = self.E_0.detach()
        
    def get_E_final(self):
        
        # E_final = α_0 E(0) + α_1 E(1) + α_2 E(2) + ... + α_K E(K)
        #         = α_0 E(0) + α_1A E(0) + α_2 A^2 E(0) + ... + α_K A^K E(0)
        
        E_i = self.E_0
        E_final_sum = E_i
        
        for _ in range(0, self.K):
            E_i = torch.sparse.mm(self.A_tilde, E_i)
            E_final_sum = torch.add(E_final_sum, E_i)
            
        E_final = E_final_sum/(self.K + 1)
        self.E_final_detached = E_final.detach()
        
        return E_final
        
        

    def forward(self, user_batch, item_batch):
        
        E_final = self.get_E_final()
        
        USER_factors, ITEM_factors = torch.split(E_final, [self.n_users, self.n_items])
        
        prediction = torch.einsum("bk,bk->b", USER_factors[user_batch,:], ITEM_factors[item_batch,:])
        
        return prediction

    
    def get_numpy_E_final(self):

        USER_factors, ITEM_factors = torch.split(self.E_final_detached, [self.n_users, self.n_items])
        USER_factors = USER_factors.detach().cpu().numpy()
        ITEM_factors = ITEM_factors.detach().cpu().numpy()
        
        return USER_factors, ITEM_factors



class _LightGCNRecommender(BaseMatrixFactorizationRecommender, Incremental_Training_Early_Stopping):
    """ LightGCNRecommender

    Consider the adjacency matrix   A = |  0     URM |
                                        | URM.T   0  |

    LightGCN learns user and item embeddings E based on a convolution over the graph, the model in matrix form is
    Given D the diagonal matrix containing the node degree, the embeddings are computed as:
    A_tilde = D^(-1/2)*A*D^(-1/2)
    E^(0) = randomly initialized
    E^(k+1) = A_tilde*E^(k)

    E_final = alpha_0 * E^(0) + alpha_1 * E^(1) ... alpha_k * E^(k)
            = alpha_0 * E^(0) + alpha_1 * A_tilde * E^(0) ... alpha_k * A_tilde^k * E^(0)

    In LightGCN E^(0) is trained and alpha can be optimized and learned, but the paper states
    a good predefined value is 1/(K+1)

    @inproceedings{DBLP:conf/sigir/0001DWLZ020,
      author    = {Xiangnan He and
                   Kuan Deng and
                   Xiang Wang and
                   Yan Li and
                   Yong{-}Dong Zhang and
                   Meng Wang},
      editor    = {Jimmy X. Huang and
                   Yi Chang and
                   Xueqi Cheng and
                   Jaap Kamps and
                   Vanessa Murdock and
                   Ji{-}Rong Wen and
                   Yiqun Liu},
      title     = {LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation},
      booktitle = {Proceedings of the 43rd International {ACM} {SIGIR} conference on
                   research and development in Information Retrieval, {SIGIR} 2020, Virtual
                   Event, China, July 25-30, 2020},
      pages     = {639--648},
      publisher = {{ACM}},
      year      = {2020},
      url       = {https://doi.org/10.1145/3397271.3401063},
      doi       = {10.1145/3397271.3401063},
      timestamp = {Wed, 03 Aug 2022 15:48:33 +0200},
      biburl    = {https://dblp.org/rec/conf/sigir/0001DWLZ020.bib},
      bibsource = {dblp computer science bibliography, https://dblp.org}
    }


    """

    RECOMMENDER_NAME = "_LightGCNRecommender"

    def __init__(self, URM_train, use_gpu = True, verbose = True):
        super(_LightGCNRecommender, self).__init__(URM_train, verbose = verbose)
        
        self._data_iterator_class = None
        self._loss_function = None

        if use_gpu:
            assert torch.cuda.is_available(), "GPU is requested but not available"
            self.device = torch.device("cuda:0")
            torch.cuda.empty_cache()
        else:
            self.device = torch.device("cpu:0")


    def fit(self,
            epochs = 300,
            K = 5,
            batch_size = 64,
            embedding_size = 100,
            l2_reg = 1e-4,
            sgd_mode = 'adam',
            learning_rate = 1e-2,
            **earlystopping_kwargs):
        
        self.l2_reg = l2_reg

        self._data_iterator = self._data_iterator_class(self.URM_train, batch_size = batch_size)

        torch.cuda.empty_cache()
        
        self._model = _LightGCNModel(self.URM_train,
                                     K,
                                     embedding_size,
                                     self.device)

        self._optimizer = get_optimizer(sgd_mode.lower(), self._model, learning_rate, self.l2_reg*learning_rate)
        
        # Initializing for epoch 0
        self._prepare_model_for_validation()
        self._update_best_model()
        
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
    
        self._train_with_early_stopping(epochs,
                                        algorithm_name = self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))
        # prof.export_chrome_trace("trace.json")
        
        self._print("Training complete")

        self.USER_factors = self.USER_factors_best.copy()
        self.ITEM_factors = self.ITEM_factors_best.copy()


    def _prepare_model_for_validation(self):
        self.USER_factors, self.ITEM_factors = self._model.get_numpy_E_final()


    def _update_best_model(self):
        self.USER_factors_best = self.USER_factors.copy()
        self.ITEM_factors_best = self.ITEM_factors.copy()


    def _run_epoch(self, num_epoch):

        if self.verbose:
            batch_iterator = tqdm(self._data_iterator)
        else:
            batch_iterator = self._data_iterator

        epoch_loss = 0
        for batch in batch_iterator:
            
            # Clear previously computed gradients
            self._optimizer.zero_grad()
    
            # Compute the loss function of the current batch
            loss = self._loss_function(self._model, batch, self.device)
    
            # Compute gradients given current loss
            loss.backward()
            epoch_loss += loss.item()
    
            # Apply gradient using the selected optimizer
            self._optimizer.step()
            
        self._print("Loss {:.2E}".format(epoch_loss))
        




class LightGCN_BPR_Recommender(_LightGCNRecommender):

    RECOMMENDER_NAME = "LightGCN_BPR_Recommender"

    def __init__(self, URM_train, use_gpu = True, use_cython_sampler = True, verbose = True):
        super(LightGCN_BPR_Recommender, self).__init__(URM_train, use_gpu = use_gpu, verbose = verbose)

        self._data_iterator_class = BPRIterator_cython if use_cython_sampler else BPRIterator
        self._loss_function = loss_BPR



class LightGCN_softplus_Recommender(_LightGCNRecommender):

    RECOMMENDER_NAME = "LightGCN_softplus_Recommender"

    def __init__(self, URM_train, use_gpu = True, use_cython_sampler = True, verbose = True):
        super(LightGCN_softplus_Recommender, self).__init__(URM_train, use_gpu = use_gpu, verbose = verbose)

        self._data_iterator_class = InteractionAndNegativeIterator_cython if use_cython_sampler else InteractionAndNegativeIterator
        self._loss_function = loss_softplus


class LightGCN_MSE_Recommender(_LightGCNRecommender):

    RECOMMENDER_NAME = "LightGCN_MSE_Recommender"

    def __init__(self, URM_train, use_gpu = True, use_cython_sampler = True, verbose = True):
        super(LightGCN_MSE_Recommender, self).__init__(URM_train, use_gpu = use_gpu, verbose = verbose)

        self._data_iterator_class = InteractionIterator_cython if use_cython_sampler else InteractionIterator
        self._loss_function = loss_MSE


