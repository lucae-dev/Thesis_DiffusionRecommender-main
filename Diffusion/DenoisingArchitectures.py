#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 09/10/2022

@author: Maurizio Ferrari Dacrema
"""

import math
from torch import nn
import torch

# Multihead attention model adjusted, you are supposed to pass a sequence of user profile 
# embeddings, in this case the sequence length is already the batch number of user profiles,
# so we could pass a batch of batches of user profiles!!! (in this case the sequence length indicates actually the dimension of the batch of user profiles, while the batch indicates the number of batches passed at the same time)
#Input to forward -> (xb, xb, xb, null) with xb: (n_of_batches, batch, emb_dim)
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model:int, h:int, dropout = None, device = None, use_gpu = True) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model% h == 0, "d_model is not divisable by h(MultiHeadAttentionBlock)"#checking d_model is divisable by number of heads
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model) #query layer
        self.w_k = nn.Linear(d_model,d_model) #key layer
        self.w_v = nn.Linear(d_model,d_model) # value layer

        self.w_o = nn.Linear(d_model,d_model) #output layer
        self.dropout = dropout
        
        self.device = device;

        self.denoising_model_loss = torch.zeros(1, requires_grad = False, device = device)


    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # (Batch, h, Seq_Len,d_k) @((Batch, h, d_k, Seq_Len,))--> (Batch, h, Seq_Len, Seq_Len )
        attention_scores = (query @ key.transpose(-2,-1)/math.sqrt(d_k))
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) 
        if dropout is not None:
           attention_scores = dropout(attention_scores)

        return (attention_scores@value), attention_scores

        #modified to accept as input a batch of user profiles noised emb and return a batch of denoised user profiles emb
        #mask avoids some input to see at others (for sequence models is useful to not make previous words the future ones)
    def forward(self, x, mask=None):
        x_unsqueezed = torch.unsqueeze(x, 0)

        query = self.w_q(x_unsqueezed) #(Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        key = self.w_k(x_unsqueezed) #(Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        value = self.w_v(x_unsqueezed) #(Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)

        #here we basically divide the 'embeddings' of dimension domodel in h parts of dimension d_k
        #(Batch, Seq_Len, d_model)-- >(Batch,Seq_Len, h, d_k) --transpose-->(Batch,h,Seq_Len,d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2) #transpose switches the second(1) and third (2)dimension
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x,self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        #(Batch, h, Seq_Len, d_k) -- > (Batch, Seq_Len, h, d_k) --> (Batch,Seq_Len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        
        return torch.squeeze(self.w_o(x),0)
    
    def loss(self):
        return self.denoising_model_loss

#encoder architecture is the list of dimensions of the layers that starts with the input size, arrive at a bottleneck (encoded features) and then goes back up to input dimension(decoded features, """restoring""" the input)
class AutoencoderModel(nn.Module):

    def __init__(self, encoder_architecture = None, activation_function = torch.nn.ReLU, use_batch_norm = False, use_dropout = False, dropout_p = 0.3, device=None):
        super().__init__()

        self._activation_function = activation_function

        assert encoder_architecture is not list, "encoder_architecture must be a list"

        for i in range(len(encoder_architecture)):
            assert encoder_architecture[i] > 0, "encoder_architecture must be a list of values > 0"

        for i in range(len(encoder_architecture)-1):
            assert encoder_architecture[i] > encoder_architecture[i+1], "encoder_architecture must be a list of decreasing values"

        self._encoder_network = torch.nn.Sequential()

    #add a module(level) of the appriotrate dimension for every layer in the architecture
        for i in range(len(encoder_architecture)-1):
            in_features = encoder_architecture[i]
            out_features = encoder_architecture[i+1]
            if use_dropout:
                self._encoder_network.add_module("dropout_{}".format(i), nn.Dropout(p=dropout_p))
            self._encoder_network.add_module("layer_{}".format(i), nn.Linear(in_features = in_features, out_features = out_features))
            if use_batch_norm:
                self._encoder_network.add_module("batch_norm_{}".format(i), nn.BatchNorm1d(out_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
            self._encoder_network.add_module("activation_{}".format(i), self._activation_function())

        self._decoder_network = torch.nn.Sequential()
        decoder_architecture = encoder_architecture[::-1]

        for i in range(len(decoder_architecture)-1):
            in_features = decoder_architecture[i]
            out_features = decoder_architecture[i+1]
            if use_dropout:
                self._decoder_network.add_module("dropout_{}".format(i), nn.Dropout(p=dropout_p))
            self._decoder_network.add_module("layer_{}".format(i), nn.Linear(in_features = in_features, out_features = out_features))
            if use_batch_norm:
                self._decoder_network.add_module("batch_norm_{}".format(i), nn.BatchNorm1d(out_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
            self._decoder_network.add_module("activation_{}".format(i), self._activation_function())


        self.denoising_model_loss = torch.zeros(1, requires_grad=False, device=device)

    def encode(self, layer_input):
        return self._encoder_network(layer_input)

    def decode(self, encoding):
        return self._decoder_network(encoding)

    def forward(self, layer_input):
        encoding = self.encode(layer_input)
        return self._decoder_network(encoding)

    def loss(self):
        return self.denoising_model_loss


class ResidualAutoencoderModel(nn.Module):

    def __init__(self, encoder_architecture = None, activation_function = torch.nn.ReLU, use_batch_norm = False, use_dropout = False, dropout_p = 0.3, device=None):
        super().__init__()

        raise NotImplementedError

        self._activation_function = activation_function

        assert encoder_architecture is not list, "encoder_architecture must be a list"

        for i in range(len(encoder_architecture)):
            assert encoder_architecture[i] > 0, "encoder_architecture must be a list of values > 0"

        for i in range(len(encoder_architecture)-1):
            assert encoder_architecture[i] > encoder_architecture[i+1], "encoder_architecture must be a list of decreasing values"

        self._encoder_network = torch.nn.Sequential()
        self._decoder_network = torch.nn.Sequential()

        decoder_architecture = encoder_architecture[::-1]

        for i in range(len(encoder_architecture)-1):
            in_features = encoder_architecture[i]
            out_features = encoder_architecture[i+1]
            if use_dropout:
                self._encoder_network.add_module("dropout_{}".format(i), nn.Dropout(p=dropout_p))
            self._encoder_network.add_module("layer_{}".format(i), nn.Linear(in_features = in_features, out_features = out_features))
            if use_batch_norm:
                self._encoder_network.add_module("batch_norm_{}".format(i), nn.BatchNorm1d(out_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
            self._encoder_network.add_module("activation_{}".format(i), self._activation_function())

            in_features = decoder_architecture[i]
            out_features = decoder_architecture[i+1]
            if use_dropout:
                self._decoder_network.add_module("dropout_{}".format(i), nn.Dropout(p=dropout_p))
            self._decoder_network.add_module("layer_{}".format(i), nn.Linear(in_features = out_features, out_features = in_features))
            if use_batch_norm:
                self._decoder_network.add_module("batch_norm_{}".format(i), nn.BatchNorm1d(in_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
            # self._encoder_network._modules["layer_{}".format(i)]
            self._decoder_network.add_module("activation_{}".format(i), self._activation_function())

        #
        #
        # for i in range(len(encoder_architecture)-1):
        #     in_features = encoder_architecture[i]
        #     out_features = encoder_architecture[i+1]
        #     if use_dropout:
        #         self._encoder_network.add_module("dropout_{}".format(i), nn.Dropout(p=dropout_p))
        #     self._encoder_network.add_module("layer_{}".format(i), nn.Linear(in_features = in_features, out_features = out_features))
        #     if use_batch_norm:
        #         self._encoder_network.add_module("batch_norm_{}".format(i), nn.BatchNorm1d(out_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        #     self._encoder_network.add_module("activation_{}".format(i), self._activation_function())
        #
        # self._decoder_network = torch.nn.Sequential()
        # decoder_architecture = encoder_architecture[::-1]
        #
        # for i in range(len(decoder_architecture)-1):
        #     in_features = decoder_architecture[i]
        #     out_features = decoder_architecture[i+1]
        #     if use_dropout:
        #         self._decoder_network.add_module("dropout_{}".format(i), nn.Dropout(p=dropout_p))
        #     self._decoder_network.add_module("layer_{}".format(i), nn.Linear(in_features = in_features, out_features = out_features))
        #     if use_batch_norm:
        #         self._decoder_network.add_module("batch_norm_{}".format(i), nn.BatchNorm1d(out_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        #     self._decoder_network.add_module("activation_{}".format(i), self._activation_function())


        self.denoising_model_loss = torch.zeros(1, requires_grad=False, device=device)

    def encode(self, layer_input):
        return self._encoder_network(layer_input)

    def decode(self, encoding):
        return self._decoder_network(encoding)

    def forward(self, layer_input):
        encoding = self.encode(layer_input)
        return self._decoder_network(encoding)

    def loss(self):
        return self.denoising_model_loss



class VariationalAutoencoderModel(nn.Module):

    def __init__(self, encoder_architecture = None, activation_function = torch.nn.ReLU, use_batch_norm = False, use_dropout = False, dropout_p = 0.3, device=None):
        super().__init__()

        self._activation_function = activation_function

        assert encoder_architecture is not list, "encoder_architecture must be a list"

        for i in range(len(encoder_architecture)):
            assert encoder_architecture[i] > 0, "encoder_architecture must be a list of values > 0"

        for i in range(len(encoder_architecture)-1):
            assert encoder_architecture[i] > encoder_architecture[i+1], "encoder_architecture must be a list of decreasing values"

        self._partial_encoder_network = torch.nn.Sequential()

        for i in range(len(encoder_architecture)-2):
            in_features = encoder_architecture[i]
            out_features = encoder_architecture[i+1]
            if use_dropout:
                self._partial_encoder_network.add_module("dropout_{}".format(i), nn.Dropout(p=dropout_p))
            self._partial_encoder_network.add_module("layer_{}".format(i), nn.Linear(in_features = in_features, out_features = out_features))
            if use_batch_norm:
                self._partial_encoder_network.add_module("batch_norm_{}".format(i), nn.BatchNorm1d(out_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
            self._partial_encoder_network.add_module("activation_{}".format(i), self._activation_function())

        in_features = encoder_architecture[-2]
        encoding_features = encoder_architecture[-1]
        self._mu = nn.Linear(in_features, encoding_features)
        self._sigma = nn.Linear(in_features, encoding_features)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.N.loc = self.N.loc.to(device) # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(device)
        self.kl_loss = 0

        self._decoder_network = torch.nn.Sequential()
        decoder_architecture = encoder_architecture[::-1]

        for i in range(len(decoder_architecture)-1):
            in_features = decoder_architecture[i]
            out_features = decoder_architecture[i+1]
            if use_dropout:
                self._decoder_network.add_module("dropout_{}".format(i), nn.Dropout(p=dropout_p))
            self._decoder_network.add_module("layer_{}".format(i), nn.Linear(in_features = in_features, out_features = out_features))
            if use_batch_norm:
                self._decoder_network.add_module("batch_norm_{}".format(i), nn.BatchNorm1d(out_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
            self._decoder_network.add_module("activation_{}".format(i), self._activation_function())


    def encode(self, layer_input):
        return self._partial_encoder_network(layer_input)

    def decode(self, encoding):
        return self._decoder_network(encoding)

    def forward(self, layer_input):
        partial_encoding = self.encode(layer_input)

        mu =  self._mu(partial_encoding)
        sigma = torch.exp(self._sigma(partial_encoding))
        encoding = mu + sigma*self.N.sample(mu.shape)
        self.kl_loss = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()

        return self._decoder_network(encoding)

    def loss(self):
        return self.kl_loss


class SDenseModel(nn.Module):

    def __init__(self, n_items = None, device = None):
        super().__init__()

        self._S = torch.nn.Parameter(torch.zeros((n_items, n_items)))
        self.denoising_model_loss = torch.zeros(1, requires_grad=False, device=device)

    def forward(self, layer_input):
        # R = R*V*V.t
        # layer_output = layer_input.dot(self._embedding_item).dot(self._embedding_item.T)
        # layer_output = torch.einsum("bi,ei,ie->bi", layer_input, self._embedding_item.T, self._embedding_item)
        layer_output = torch.einsum("bi,ik->bk", layer_input, self._S)
        return layer_output

    def loss(self):
        return self.denoising_model_loss


class SNoDiagonalDenseModel(nn.Module):

    def __init__(self, n_items = None, device = None):
        super().__init__()

        self._S = torch.nn.Parameter(torch.zeros((n_items, n_items)))
        self.denoising_model_loss = torch.zeros(1, requires_grad=False, device=device)

    def forward(self, layer_input):
        # R = R*V*V.t
        # layer_output = layer_input.dot(self._embedding_item).dot(self._embedding_item.T)
        # layer_output = torch.einsum("bi,ei,ie->bi", layer_input, self._embedding_item.T, self._embedding_item)
        with torch.no_grad():
            torch.diagonal(self._S, 0).zero_()
        layer_output = torch.einsum("bi,ik->bk", layer_input, self._S)
        return layer_output

    def loss(self):
        return self.denoising_model_loss


class SNoDiagonalDenseAndAutoencoderModel(nn.Module):

    def __init__(self, encoder_architecture = None, activation_function = torch.nn.ReLU, use_batch_norm = False, use_dropout = False, dropout_p = 0.3, device = None):
        super().__init__()

        self._autoencoder = AutoencoderModel(encoder_architecture = encoder_architecture,
                                             activation_function = activation_function,
                                             use_batch_norm = use_batch_norm,
                                             use_dropout = use_dropout,
                                             dropout_p = dropout_p,
                                             device = device)

        n_items = encoder_architecture[0]

        self._S_dense = SNoDiagonalDenseModel(n_items = n_items,
                                              device = device)

    def forward(self, layer_input):
        return self._autoencoder(layer_input) + self._S_dense(layer_input)

    def loss(self):
        return self._autoencoder.denoising_model_loss + self._S_dense.denoising_model_loss


class SFactorizedModel(nn.Module):

    def __init__(self, embedding_size = None, n_items = None, device = None):
        super().__init__()

        self._embedding_item = torch.nn.Parameter(torch.randn((n_items, embedding_size)))

        self.denoising_model_loss = torch.zeros(1, requires_grad=False, device=device)

    def forward(self, layer_input):
        # input shape is batch_size x n items
        # r_hat_bi = SUM{j=0}{j=n items} r_bj S_ji
        #          = SUM{j=0}{j=n items} r_bj SUM{e=0}{e=embedding_size} V_je * V_ie
        #          = SUM{j=0}{j=n items} SUM{e=0}{e=embedding_size} r_bj * V_je * V_ie
        layer_output = torch.einsum("bj,je,ie->bi", layer_input, self._embedding_item, self._embedding_item)
        return layer_output

    def loss(self):
        return self.denoising_model_loss



class AsySVDModel(nn.Module):

    def __init__(self, embedding_size = None, n_items = None, device = None):
        super().__init__()

        self._embedding_item_1 = torch.nn.Parameter(torch.randn((n_items, embedding_size)))
        self._embedding_item_2 = torch.nn.Parameter(torch.randn((embedding_size, n_items)))

        self.denoising_model_loss = torch.zeros(1, requires_grad=False, device=device)

    def forward(self, layer_input):
        # input shape is batch_size x n items
        # r_hat_bi = SUM{e=0}{e=embedding_size} SUM{j=0}{j=n items} r_bj * V1_je * V2_ei
        layer_output = torch.einsum("bj,je,ei->bi", layer_input, self._embedding_item_1, self._embedding_item_2)
        return layer_output

    def loss(self):
        return self.denoising_model_loss



import time, sys
import numpy as np
import scipy.sparse as sps
from sklearn.utils.extmath import randomized_svd

class ItemSVDModel(nn.Module):

    def __init__(self, embedding_size = None, URM_train = None, device = None):
        super().__init__()

        # R = U * Sigma * V.t
        # S = R.t * R = V * Sigma * U.t * U * Sigma * V.t = V * Sigma^2 * V.t
        _, Sigma, Vt = randomized_svd(URM_train,
                                     n_components = embedding_size,
                                     random_state = None)

        # self._singular_vector_user = torch.tensor(U, requires_grad=False)
        self._V = torch.tensor(Vt.T, requires_grad=False, device=device)
        self._Sigma = torch.nn.Parameter(torch.tensor(Sigma**2))

        self.denoising_model_loss = torch.zeros(1, requires_grad=False, device=device)

    def forward(self, layer_input):
        # input shape is batch_size x n items
        # r_hat_bi = SUM{j=0}{j=n items} r_bj S_ji
        #          = SUM{j=0}{j=n items} r_bj SUM{e=0}{e=embedding_size} V_je * Sigma_e * V_ie
        layer_output = torch.einsum("bj,je,e,ie->bi", layer_input, self._V, self._Sigma, self._V)
        return layer_output

    def loss(self):
        return self.denoising_model_loss



class TowerModel(nn.Module):

    def __init__(self, tower_architecture = None, activation_function = torch.nn.ReLU, use_batch_norm = True, use_dropout = True, dropout_p = 0.3, device = None):
        super().__init__()

        self._activation_function = activation_function

        assert tower_architecture is not list, "encoder_architecture must be a list"

        for i in range(len(tower_architecture)):
            assert tower_architecture[i] > 0, "encoder_architecture must be a list of values > 0"

        self._tower_network = torch.nn.Sequential()

        for i in range(len(tower_architecture)-1):
            in_features = tower_architecture[i]
            out_features = tower_architecture[i+1]
            if use_dropout:
                self._tower_network.add_module("dropout_{}".format(i), nn.Dropout(p=dropout_p))
            self._tower_network.add_module("layer_{}".format(i), nn.Linear(in_features = in_features, out_features = out_features))
            if use_batch_norm:
                self._tower_network.add_module("batch_norm_{}".format(i), nn.BatchNorm1d(out_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
            self._tower_network.add_module("activation_{}".format(i), self._activation_function())

        self.denoising_model_loss = torch.zeros(1, requires_grad=False, device=device)


    def forward(self, layer_input):
        return self._tower_network(layer_input)

    def loss(self):
        return self.denoising_model_loss