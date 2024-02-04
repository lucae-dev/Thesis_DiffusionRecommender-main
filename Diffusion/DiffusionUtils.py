#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 09/10/2022

@author: Maurizio Ferrari Dacrema
"""

from torch.utils.data import Dataset
import scipy.sparse as sps
import torch
from Evaluation.Evaluator import get_result_string_df
import pandas as pd

class UserProfile_Dataset(Dataset):
    def __init__(self, URM_train, device):
        super().__init__()
        URM_train = sps.csr_matrix(URM_train)
        self.device = device

        self.n_users, self.n_items = URM_train.shape
        self._indptr = URM_train.indptr
        self._indices = torch.tensor(URM_train.indices, dtype = torch.long, device=device)
        self._data = torch.tensor(URM_train.data, dtype = torch.float, device=device)

    def __len__(self):
        return self.n_users

    def __getitem__(self, user_id):
        start_pos = self._indptr[user_id]
        end_pos = self._indptr[user_id+1]

        user_profile = torch.zeros(self.n_items, dtype=torch.float, requires_grad=False, device=self.device)
        user_profile[self._indices[start_pos:end_pos]] = self._data[start_pos:end_pos]

        return user_profile


    def get_batch(self, user_id_array):

        start_pos = self._indptr[user_id_array]
        end_pos = self._indptr[user_id_array+1]

        user_profile = torch.zeros((len(user_id_array), self.n_items), dtype=torch.float, requires_grad=False)
        for i in range(len(user_id_array)):
            user_profile[i,self._indices[start_pos[i]:end_pos[i]]] = self._data[start_pos[i]:end_pos[i]]

        return user_profile





class EvaluatorTrainingLoss(object):

    def __init__(self, loss_attribute_name, n_users, minimize = False):

        self._loss_attribute_name = loss_attribute_name
        self._minimize = minimize
        self._n_users = n_users

    def evaluateRecommender(self, recommender_object):

        loss_value = getattr(recommender_object, self._loss_attribute_name)
        loss_value = -1*loss_value if self._minimize else loss_value

        results_df = pd.DataFrame(columns=[self._loss_attribute_name], index=[self._loss_attribute_name])
        results_df[self._loss_attribute_name] = loss_value

        results_run_string = get_result_string_df(results_df)

        return results_df, results_run_string





