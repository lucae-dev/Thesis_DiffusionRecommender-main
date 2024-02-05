#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 29/09/2022

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.DataIO import DataIO

import os, shutil, copy, math
import torch.nn.functional as F
from tqdm.auto import tqdm

from Diffusion.DenoisingArchitectures import *
from Diffusion.PositionalEncoding import SinusoidalPositionalEncoding
from Diffusion.NoiseSchedule import LinearNoiseSchedule
from Diffusion.DiffusionUtils import UserProfile_Dataset
from Diffusion.architecture_utils import generate_autoencoder_architecture, generate_tower_architecture

def _get_optimizer(optimizer_label, model, learning_rate, l2_reg):

    if optimizer_label.lower() == "adagrad":
        return torch.optim.Adagrad(model.parameters(), lr = learning_rate, weight_decay = l2_reg*learning_rate)
    elif optimizer_label.lower() == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr = learning_rate, weight_decay = l2_reg*learning_rate)
    elif optimizer_label.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = l2_reg*learning_rate)
    elif optimizer_label.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr = learning_rate, weight_decay = l2_reg*learning_rate)
    else:
        raise ValueError("sgd_mode attribute value not recognized.")


def _get_activation_function(function_name):

    function_mapper = {
        "ReLU": torch.nn.ReLU,
        "LeakyReLU": torch.nn.LeakyReLU,
        "Sigmoid": torch.nn.Sigmoid,
        "GELU": torch.nn.GELU,
        "Tanh": torch.nn.Tanh,
    }

    return function_mapper[function_name]


class _GaussianDiffusionModel(nn.Module):
    def __init__(self,
                 denoiser_model,
                 noise_schedule,
                 positional_encoding,
                 noise_timesteps = 1000,
                 # inference_timesteps = 1000,
                 # loss_type = 'l1',
                 objective = 'pred_noise',
                 # beta_schedule = None,
                 # p2_loss_weight_gamma = 0.,  # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
                 # p2_loss_weight_k = 1,
                 # ddim_sampling_eta = 1.
                 ):
        super().__init__()
        # assert not (type(self) == GaussianDiffusionModel and denoiser_model.channels != denoiser_model.out_dim)

        self.denoiser_model = denoiser_model
        self.noise_schedule = noise_schedule
        self.positional_encoding = positional_encoding
        self.noise_timesteps = noise_timesteps
        # self.inference_timesteps = inference_timesteps
        # self.loss_type = loss_type

        assert objective in {'pred_noise', 'pred_x0'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start)'
        self.objective = objective


    def q_sample(self, x_start, t, gaussian_noise = None):
        """
        Apply noise to the sample by calculating the total noise that would be applied after t steps.
        The sum of t gaussian noise steps is also a gaussian so we can avoid to iterate t steps and calculate
        directly the final one.

        noise at t = N(sqrt (1-alpha)*x_start, (1-alpha)*I )
        alpha = prod{s=1 to t} (1-beta_t)

        :param x_start:
        :param t:
        :param gaussian_noise:
        :return:
        """

        a_signed = self.noise_schedule.get_a_signed(t)
        print("a_signed shape: ")
        print(a_signed.shape)
        batch_size = len(t)
        print("a_sized_reshaped: ")
        print(a_signed.reshape(batch_size,1).shape)
        x_noisy = torch.sqrt(a_signed).reshape(batch_size, 1)*x_start + torch.sqrt(1 - a_signed).reshape(batch_size, 1) * gaussian_noise

        return x_noisy

    def forward(self, x_start_batch, t):
        """

        :param x_start:
        :param t:
        :return:
        """

        gaussian_noise = torch.randn_like(x_start_batch)

        # noise sample
        x_noisy = self.q_sample(x_start = x_start_batch, t = t, gaussian_noise = gaussian_noise)
        x_noisy = x_noisy + self.positional_encoding.get_encoding(t)

        denoiser_prediction = self.denoiser_model(x_noisy)
        denosier_loss = self.denoiser_model.loss()

        if self.objective == 'pred_noise':
            target = gaussian_noise
        elif self.objective == 'pred_x0':
            target = x_start_batch
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss_batch = F.mse_loss(denoiser_prediction, target) + denosier_loss

        return loss_batch

    def sample_from_user_profile(self, user_profile, inference_timesteps):
        """
        # TODO check math
        :param user_profile:
        :param inference_timesteps:
        :return:
        """

        with torch.no_grad():
            gaussian_noise = torch.randn_like(user_profile, device=user_profile.device)

            user_profile_inference = self.q_sample(x_start = user_profile, t = torch.tensor([inference_timesteps], device=user_profile.device, dtype=torch.long), gaussian_noise = gaussian_noise)

            for timestep in range(inference_timesteps, 0, -1):
                user_profile_inference = user_profile_inference + self.positional_encoding.get_encoding(torch.tensor([timestep], device=user_profile.device))

                user_profile_inference = self.denoiser_model.forward(user_profile_inference)

                posterior_mean = user_profile * self.noise_schedule._posterior_mean_c_x_start[timestep] + user_profile_inference * self.noise_schedule._posterior_mean_c_x_t[timestep]

                if timestep > 0:
                    noise = torch.randn_like(user_profile, device = user_profile.device)
                    user_profile_inference = posterior_mean + (0.5*self.noise_schedule._posterior_log_variance_clipped[timestep]).exp() * noise
                else:
                    user_profile_inference = posterior_mean

        return user_profile_inference




class SimpleAttentionDiffusionRecommender(BaseRecommender, Incremental_Training_Early_Stopping):
    """
    Diffusion model based on user profiles, using a single self-attention layer as denoising architecture
    """

    RECOMMENDER_NAME = "SimpleAttentionDiffusionRecommender"

    def __init__(self, URM_train, use_gpu = True, verbose = True):
        super(SimpleAttentionDiffusionRecommender, self).__init__(URM_train, verbose = verbose)

        if use_gpu:
            # Check for CUDA availability (for NVIDIA GPUs)
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                torch.cuda.empty_cache()
            # Check for MPS availability (for Apple Silicon GPUs)
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                raise RuntimeError("GPU is requested but neither CUDA nor MPS is available")
        else:
            self.device = torch.device("cpu:0")
            print("no GPU")

        self.warm_user_ids = np.arange(0, self.n_users)[np.ediff1d(sps.csr_matrix(self.URM_train).indptr) > 0]

    def _set_inference_timesteps(self,inference_timesteps):
        self.inference_timesteps = inference_timesteps
        

    def _compute_item_score(self, user_id_array, items_to_compute = None):
        """
    
        :param user_id_array:
        :param items_to_compute:
        :return:
        """
    
        # user_profile_batch = self._dataset.get_batch(user_id_array).to(self.device)
        user_profile_batch = self.URM_train[user_id_array]
        user_profile_batch = torch.sparse_csr_tensor(user_profile_batch.indptr,
                                                    user_profile_batch.indices,
                                                    user_profile_batch.data,
                                                    size=user_profile_batch.shape,
                                                    dtype=torch.float32,
                                                    device=self.device,
                                                    requires_grad=False).to_dense()

        user_profile_inference = self._model.sample_from_user_profile(user_profile_batch, self.inference_timesteps).cpu().detach().numpy()

        if items_to_compute is None:
            item_scores = user_profile_inference
        else:
            item_scores = - np.ones((len(user_id_array), self.n_items), dtype=np.float32)*np.inf

        return item_scores



    def _init_model(self):

        positional_encoding = SinusoidalPositionalEncoding(embedding_size = self.n_items, device=self.device)
        noise_schedule = LinearNoiseSchedule(start_beta = self.start_beta, end_beta = self.end_beta,
                                             device=self.device, noise_timesteps = self.noise_timesteps)

        self.denoiser_model = MultiHeadAttentionBlock(d_model = self.embeddings_dim, h = self.heads, device = self.device, dropout=None)

        self._model = _GaussianDiffusionModel(self.denoiser_model,
                                              noise_schedule,
                                              positional_encoding,
                                              noise_timesteps = self.noise_timesteps,
                                              # inference_timesteps = self.inference_timesteps,
                                              # loss_type = 'l1',
                                              objective = self.objective,
                                              ).to(self.device)
        
        self.encoder_model = SimpleAutoencoder(self.n_items, self.embeddings_dim, device=self.device)


    def fit(self, epochs = 300,
            batch_size = 128,
            embeddings_dim = 128,
            heads = 1,
            l2_reg = 1e-4,
            sgd_mode = 'adam',
            learning_rate = 1e-2,
            noise_timesteps = 1000,
            inference_timesteps = 500,
            show_progress_bar = False,
            start_beta = 0.0001,
            end_beta = 0.10,
            activation_function = "ReLU",
            use_batch_norm = False,
            use_dropout = False,
            dropout_p = 0.3,
            objective = "pred_noise",
            **earlystopping_kwargs):


        self.activation_function = activation_function
        self.inference_timesteps = inference_timesteps
        self.noise_timesteps = noise_timesteps
        self.batch_size = batch_size
        self.sgd_mode = sgd_mode
        self.objective = objective
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.show_progress_bar = show_progress_bar
        self.start_beta = start_beta
        self.end_beta = end_beta
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.dropout_p = dropout_p
        self.current_epoch_training_loss = 0
        self.heads = heads
        self.embeddings_dim = embeddings_dim

        self._init_model()

        self._optimizer = _get_optimizer(self.sgd_mode.lower(), self._model, self.learning_rate, self.l2_reg)

        self._update_best_model()

        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        self.loss_list = []

        self._train_with_early_stopping(epochs,
                                        algorithm_name = self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))
        # prof.export_chrome_trace("trace.json")

        self._print("Training complete")

        self._model.load_state_dict(self._model_best)

        # Set model in evaluation mode (disables dropout, for example)
        self._model.eval()


    def _prepare_model_for_validation(self):
        self._model.eval() # Particularly important if we use dropout or batch norm layers


    def _update_best_model(self):
        self._model_best = copy.deepcopy(self._model.state_dict())

    def forward(self, x_start_batch, t):
        """

        :param x_start:
        :param t:
        :return:
        """

        x_emb_batch = self.encoder_model.encode(x_start_batch)

        gaussian_noise = torch.randn_like(x_emb_batch)

        # noise sample
        x_noisy = self._model.q_sample(x_start = x_emb_batch, t = t, gaussian_noise = gaussian_noise)
        #x_noisy = x_noisy + self.positional_encoding.get_encoding(t)
        
        denoiser_prediction = self.denoiser_model.forward(x_noisy, None)
        denosier_loss = self.denoiser_model.loss()

        if self.objective == 'pred_noise':
            target = gaussian_noise
        elif self.objective == 'pred_x0':
            target = x_emb_batch
        else:
            raise ValueError(f'unknown objective {self.objective}')

        decoded_prediction = self.encoder_model.decode(denoiser_prediction)

        loss_batch = F.mse_loss(decoded_prediction, x_start_batch) + F.mse_loss(denoiser_prediction, target) + denosier_loss

        return loss_batch

    def _run_epoch(self, num_epoch):

        self.current_epoch_training_loss = 0
        self._model.train()

        #batches must become number of batches, the batch size will another parameter -> add parameter n_batches
        num_batches_per_epoch = math.ceil(len(self.warm_user_ids) / self.batch_size)

        iterator = tqdm(range(num_batches_per_epoch)) if self.show_progress_bar else range(num_batches_per_epoch)

        for _ in iterator:

            user_batch = torch.LongTensor(np.random.choice(self.warm_user_ids, size=self.batch_size))

            # Transferring only the sparse structure to reduce the data transfer
            user_batch_tensor = self.URM_train[user_batch]
            user_batch_tensor = torch.sparse_csr_tensor(user_batch_tensor.indptr,
                                                        user_batch_tensor.indices,
                                                        user_batch_tensor.data,
                                                        size=user_batch_tensor.shape,
                                                        dtype=torch.float32,
                                                        device=self.device,
                                                        requires_grad=False).to_dense()
            
            print("user_batch_size: ")
            print(user_batch_tensor.shape)

            #here you coudl transform the user profile to reduce dimensionality before passing tbem to the forward, the forward will learn this way to predict the reduced-size-user-profile (which is like an embedding), not the normal user profile (or the noise applied to embeddign)
            # you can do it just passing the user profiles trough a linear layer, just take the encoder of an autoencoder and get it trough it

            # Clear previously computed gradients
            self._optimizer.zero_grad()

            # Sample timestamps, !!!! MIGHT HAVE TO CHANGE THE LEN??!!!
            t = torch.randint(0, self.noise_timesteps, (len(user_batch_tensor),), device=self.device, dtype=torch.long)

            # Compute prediction for each element in batch
            loss = self.forward(user_batch_tensor, t)

            # Compute gradients given current loss
            loss.backward()
            # torch.nn.utils.clip_grad_norm(self._model.parameters(), max_norm=1.0)

            # Apply gradient using the selected optimizer
            self._optimizer.step()

            self.current_epoch_training_loss += loss.item()


        self._print("Epoch {}, loss {:.2E}".format(num_epoch, self.current_epoch_training_loss))
        # self.loss_list.append(self.current_epoch_training_loss)


    def save_model(self, folder_path, file_name = None, create_zip = True):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        # Save session within a temp folder called as the desired file_name
        if not os.path.isdir(folder_path + file_name + "/.torch_state_dict"):
            os.makedirs(folder_path + file_name + "/.torch_state_dict")

        torch.save(self._model.state_dict(), folder_path + file_name + "/.torch_state_dict/torch_state_dict")

        data_dict_to_save = {
            "encoder_architecture": self.encoder_architecture,
            "activation_function": self.activation_function,
            "start_beta": self.start_beta,
            "end_beta": self.end_beta,
            "noise_timesteps": self.noise_timesteps,
            "objective": self.objective,

            "batch_size": self.batch_size,
            "sgd_mode": self.sgd_mode,
            "learning_rate": self.learning_rate,
            "l2_reg": self.l2_reg,
        }


        dataIO = DataIO(folder_path=folder_path + file_name + "/")
        dataIO.save_data(file_name="fit_attributes", data_dict_to_save = data_dict_to_save)

        # Create a zip folder containing fit_attributes and saved session
        if create_zip:
            # Unfortunately I cannot avoid compression so it is too slow for earlystopping
            shutil.make_archive(
              folder_path + file_name,          # name of the file to create
              'zip',                            # archive format - or tar, bztar, gztar
              root_dir = folder_path + file_name + "/",     # root for archive
              base_dir = None)                  # start archiving from the root_dir

            shutil.rmtree(folder_path + file_name + "/", ignore_errors=True)

        self._print("Saving complete")

    def load_model(self, folder_path, file_name = None, create_zip = True, use_gpu=False):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Loading model from file '{}'".format(folder_path + file_name))

        if create_zip:
            shutil.unpack_archive(folder_path + file_name + ".zip",
                                  folder_path + file_name + "/",
                                  "zip")

        dataIO = DataIO(folder_path=folder_path + file_name + "/")
        data_dict = dataIO.load_data(file_name="fit_attributes")

        for attrib_name in data_dict.keys():
             self.__setattr__(attrib_name, data_dict[attrib_name])

        self.use_batch_norm=False #+
        self.use_dropout=False #+
        self.dropout_p = 0.3 #+
        self._init_model()

        map_location = torch.device('cuda') if use_gpu and torch.cuda.is_available() else torch.device('cpu')
        self._model.load_state_dict(torch.load(folder_path + file_name + "/.torch_state_dict/torch_state_dict", map_location=map_location))
        self._model.eval()

        shutil.rmtree(folder_path + file_name + "/", ignore_errors=True)

        self._print("Loading complete")




class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, device = None):
        super(SimpleAutoencoder, self).__init__()

        self.device = device

        intermediate_dim = round((input_dim + encoding_dim) / 2)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),  
            nn.ReLU(),
            nn.Linear(intermediate_dim, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, input_dim),
            nn.Sigmoid() 
        )

        self.to(self.device)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, encoding):
        return self.decoder(encoding)
    
    def forward(self, x):
        encoding = self.encode(x)
        return self.decode(encoding)


"""
END OF ATTENTION
"""





class DiffusionRecommender(BaseRecommender, Incremental_Training_Early_Stopping):
    """
    Diffusion model based on the user profile, which uses an autoencoder as denoising model
    """

    RECOMMENDER_NAME = "DiffusionRecommender"


    def __init__(self, URM_train, use_gpu = True, verbose = True):
        super(DiffusionRecommender, self).__init__(URM_train, verbose = verbose)

        if use_gpu:
            assert torch.cuda.is_available(), "GPU is requested but not available"
            self.device = torch.device("cuda:0")
            torch.cuda.empty_cache()
        else:
            self.device = torch.device("cpu:0")

        self.warm_user_ids = np.arange(0, self.n_users)[np.ediff1d(sps.csr_matrix(self.URM_train).indptr) > 0]

        """
    def _compute_item_score(self, user_id_array, items_to_compute = None):
        """
        """
        :param user_id_array:
        :param items_to_compute:
        :return:
        """
        """


        # Calculate the number of subarrays needed
        num_subarrays = (len(user_id_array) + 99) // 100

        # Split the user_id_array into subarrays
        subarrays = [user_id_array[i * 100:(i + 1) * 100] for i in range(num_subarrays)]

        # Create an empty list to collect the results
        user_profile_inference_list = []

        # Perform the operations on each subarray
        for subarray in subarrays:
            # Get the user_profile_batch
            subarray = np.array(subarray)
            user_profile_batch = self.URM_train[subarray]

            # Convert user_profile_batch to dense tensor
            user_profile_batch = torch.sparse_csr_tensor(
                user_profile_batch.indptr,
                user_profile_batch.indices,
                user_profile_batch.data,
                size=user_profile_batch.shape,
                dtype=torch.float32,
                device=self.device,
                requires_grad=False
            ).to_dense()

            # Perform the model inference
            user_profile_inference = self._model.sample_from_user_profile(
                user_profile_batch,
                self.inference_timesteps
            ).cpu().detach().numpy()

            # Append the results to the list
            user_profile_inference_list.append(user_profile_inference)

        # Concatenate the results into a single array
        user_profile_inference_all = np.concatenate(user_profile_inference_list, axis=0)

        if items_to_compute is None:
            item_scores = user_profile_inference
        else:
            item_scores = - np.ones((len(user_id_array), self.n_items), dtype=np.float32)*np.inf
            item_scores[:, items_to_compute] = user_profile_inference[items_to_compute]

        # for user_index, user_id in enumerate(user_id_array):
        #     user_profile = self._dataset[user_id].to(self.device)
        #     user_profile_inference = self._model.sample_from_user_profile(user_profile, self.inference_timesteps)
        #
        #     if items_to_compute is None:
        #         item_scores[user_index,:] = user_profile_inference.cpu().detach().numpy()
        #     else:
        #         item_scores[user_index, items_to_compute] = user_profile_inference.cpu().detach().numpy()[items_to_compute]
        print("!!!!!!!!!!!!!")
        print(len(user_id_array))
        print(len(item_scores))
        print (len(item_scores)-len(user_id_array))
        return item_scores
    """

    def _set_inference_timesteps(self,inference_timesteps):
        self.inference_timesteps = inference_timesteps
        

    def _compute_item_score(self, user_id_array, items_to_compute = None):
        """
    
        :param user_id_array:
        :param items_to_compute:
        :return:
        """
    
        

        # user_profile_batch = self._dataset.get_batch(user_id_array).to(self.device)
        user_profile_batch = self.URM_train[user_id_array]
        user_profile_batch = torch.sparse_csr_tensor(user_profile_batch.indptr,
                                                    user_profile_batch.indices,
                                                    user_profile_batch.data,
                                                    size=user_profile_batch.shape,
                                                    dtype=torch.float32,
                                                    device=self.device,
                                                    requires_grad=False).to_dense()

        user_profile_inference = self._model.sample_from_user_profile(user_profile_batch, self.inference_timesteps).cpu().detach().numpy()

        if items_to_compute is None:
            item_scores = user_profile_inference
        else:
            item_scores = - np.ones((len(user_id_array), self.n_items), dtype=np.float32)*np.inf

        # for user_index, user_id in enumerate(user_id_array):
        #     user_profile = self._dataset[user_id].to(self.device)
        #     user_profile_inference = self._model.sample_from_user_profile(user_profile, self.inference_timesteps)
        #
        #     if items_to_compute is None:
        #         item_scores[user_index,:] = user_profile_inference.cpu().detach().numpy()
        #     else:
        #         item_scores[user_index, items_to_compute] = user_profile_inference.cpu().detach().numpy()[items_to_compute]


        return item_scores



    def _init_model(self):

        positional_encoding = SinusoidalPositionalEncoding(embedding_size = self.n_items, device=self.device)
        noise_schedule = LinearNoiseSchedule(start_beta = self.start_beta, end_beta = self.end_beta,
                                             device=self.device, noise_timesteps = self.noise_timesteps)

        denoiser_model = AutoencoderModel(encoder_architecture = self.encoder_architecture,
                                          activation_function = _get_activation_function(self.activation_function),
                                          use_batch_norm = self.use_batch_norm,
                                          use_dropout = self.use_dropout,
                                          dropout_p = self.dropout_p,
                                          device=self.device)

        self._model = _GaussianDiffusionModel(denoiser_model,
                                              noise_schedule,
                                              positional_encoding,
                                              noise_timesteps = self.noise_timesteps,
                                              # inference_timesteps = self.inference_timesteps,
                                              # loss_type = 'l1',
                                              objective = self.objective,
                                              ).to(self.device)


    def fit(self, epochs = 300,
            batch_size = 8,
            encoder_architecture = None,
            l2_reg = 1e-4,
            sgd_mode = 'adam',
            learning_rate = 1e-2,
            noise_timesteps = 1000,
            inference_timesteps = 500,
            show_progress_bar = False,
            start_beta = 0.0001,
            end_beta = 0.10,
            activation_function = "ReLU",
            use_batch_norm = False,
            use_dropout = False,
            dropout_p = 0.3,
            objective = "pred_noise",
            **earlystopping_kwargs):


        self.encoder_architecture = encoder_architecture
        self.activation_function = activation_function
        self.inference_timesteps = inference_timesteps
        self.noise_timesteps = noise_timesteps
        self.batch_size = batch_size
        self.sgd_mode = sgd_mode
        self.objective = objective
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.show_progress_bar = show_progress_bar
        self.start_beta = start_beta
        self.end_beta = end_beta
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.dropout_p = dropout_p
        self.current_epoch_training_loss = 0

        self._init_model()

        self._optimizer = _get_optimizer(self.sgd_mode.lower(), self._model, self.learning_rate, self.l2_reg)

        self._update_best_model()

        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        self.loss_list = []

        self._train_with_early_stopping(epochs,
                                        algorithm_name = self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))
        # prof.export_chrome_trace("trace.json")

        self._print("Training complete")

        self._model.load_state_dict(self._model_best)

        # Set model in evaluation mode (disables dropout, for example)
        self._model.eval()



    def _prepare_model_for_validation(self):
        self._model.eval() # Particularly important if we use dropout or batch norm layers


    def _update_best_model(self):
        self._model_best = copy.deepcopy(self._model.state_dict())


    def _run_epoch(self, num_epoch):

        self.current_epoch_training_loss = 0
        self._model.train()

        num_batches_per_epoch = math.ceil(len(self.warm_user_ids) / self.batch_size)
        iterator = tqdm(range(num_batches_per_epoch)) if self.show_progress_bar else range(num_batches_per_epoch)

        for _ in iterator:

            user_batch = torch.LongTensor(np.random.choice(self.warm_user_ids, size=self.batch_size))

            # Transferring only the sparse structure to reduce the data transfer
            user_batch_tensor = self.URM_train[user_batch]
            user_batch_tensor = torch.sparse_csr_tensor(user_batch_tensor.indptr,
                                                        user_batch_tensor.indices,
                                                        user_batch_tensor.data,
                                                        size=user_batch_tensor.shape,
                                                        dtype=torch.float32,
                                                        device=self.device,
                                                        requires_grad=False).to_dense()

            # Clear previously computed gradients
            self._optimizer.zero_grad()

            # Sample timestamps
            t = torch.randint(0, self.noise_timesteps, (len(user_batch_tensor),), device=self.device, dtype=torch.long)

            # Compute prediction for each element in batch
            loss = self._model.forward(user_batch_tensor, t)

            # Compute gradients given current loss
            loss.backward()
            # torch.nn.utils.clip_grad_norm(self._model.parameters(), max_norm=1.0)

            # Apply gradient using the selected optimizer
            self._optimizer.step()

            self.current_epoch_training_loss += loss.item()


        self._print("Epoch {}, loss {:.2E}".format(num_epoch, self.current_epoch_training_loss))
        # self.loss_list.append(self.current_epoch_training_loss)






    def save_model(self, folder_path, file_name = None, create_zip = True):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        # Save session within a temp folder called as the desired file_name
        if not os.path.isdir(folder_path + file_name + "/.torch_state_dict"):
            os.makedirs(folder_path + file_name + "/.torch_state_dict")

        torch.save(self._model.state_dict(), folder_path + file_name + "/.torch_state_dict/torch_state_dict")

        data_dict_to_save = {
            "encoder_architecture": self.encoder_architecture,
            "activation_function": self.activation_function,
            "start_beta": self.start_beta,
            "end_beta": self.end_beta,
            "noise_timesteps": self.noise_timesteps,
            "objective": self.objective,

            "batch_size": self.batch_size,
            "sgd_mode": self.sgd_mode,
            "learning_rate": self.learning_rate,
            "l2_reg": self.l2_reg,
        }


        dataIO = DataIO(folder_path=folder_path + file_name + "/")
        dataIO.save_data(file_name="fit_attributes", data_dict_to_save = data_dict_to_save)

        # Create a zip folder containing fit_attributes and saved session
        if create_zip:
            # Unfortunately I cannot avoid compression so it is too slow for earlystopping
            shutil.make_archive(
              folder_path + file_name,          # name of the file to create
              'zip',                            # archive format - or tar, bztar, gztar
              root_dir = folder_path + file_name + "/",     # root for archive
              base_dir = None)                  # start archiving from the root_dir

            shutil.rmtree(folder_path + file_name + "/", ignore_errors=True)

        self._print("Saving complete")





    def load_model(self, folder_path, file_name = None, create_zip = True, use_gpu=False):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Loading model from file '{}'".format(folder_path + file_name))

        if create_zip:
            shutil.unpack_archive(folder_path + file_name + ".zip",
                                  folder_path + file_name + "/",
                                  "zip")

        dataIO = DataIO(folder_path=folder_path + file_name + "/")
        data_dict = dataIO.load_data(file_name="fit_attributes")

        for attrib_name in data_dict.keys():
             self.__setattr__(attrib_name, data_dict[attrib_name])

        self.use_batch_norm=False #+
        self.use_dropout=False #+
        self.dropout_p = 0.3 #+
        self._init_model()

        map_location = torch.device('cuda') if use_gpu and torch.cuda.is_available() else torch.device('cpu')
        self._model.load_state_dict(torch.load(folder_path + file_name + "/.torch_state_dict/torch_state_dict", map_location=map_location))
        self._model.eval()

        shutil.rmtree(folder_path + file_name + "/", ignore_errors=True)

        self._print("Loading complete")






class DiffusionAutoencoderRecommender_OptimizerMask(DiffusionRecommender):
    """
    Diffusion model based on the user profile, which uses an autoencoder as denoising model

    This is the class that will be used when running an optimization as some hyperparameters are provided in a different way,
    in particular, the autoencoder architecture is parametrized using the encoding size and the layer multiplier.

    Example, if encoding = 50, next layer multiplier = 3, max hidden layers = 2
    the architecture is:
    [n_items, 450, 150, 50, 150, 450, n_items]
    """

    RECOMMENDER_NAME = "DiffusionAutoencoderRecommender"

    def fit(self, encoding_size = 50, next_layer_size_multiplier = 2, max_parameters = np.inf, max_n_hidden_layers = 3,
            noise_timesteps = 100, **kwargs):

        assert next_layer_size_multiplier > 1.0, "next_layer_size_multiplier must be > 1.0"
        assert encoding_size <= self.n_items, "encoding_size must be <= the number of items"

        encoder_architecture = generate_autoencoder_architecture(encoding_size, self.n_items, next_layer_size_multiplier, max_parameters, max_n_hidden_layers)[::-1]

        self._print("Architecture: {}".format(encoder_architecture))

        super().fit(encoder_architecture=encoder_architecture, noise_timesteps = int(noise_timesteps),
                    **kwargs)



class SNoDiagonalDenseAndAutoencoderModel_OptimizerMask(DiffusionRecommender):
    """
    Diffusion model based on the user profile, which uses the summation of autoencoder and a dense item-item similarity
    model with no diagonal as denoising model

    This is the class that will be used when running an optimization as some hyperparameters are provided in a different way,
    in particular, the autoencoder architecture is parametrized using the encoding size and the layer multiplier.

    Example, if encoding = 50, next layer multiplier = 3, max hidden layers = 2
    the architecture is:
    [n_items, 450, 150, 50, 150, 450, n_items]
    """

    RECOMMENDER_NAME = "SNoDiagonalDenseAndAutoencoderModelRecommender"

    def fit(self, encoding_size = 50, next_layer_size_multiplier = 2, max_parameters = np.inf, max_n_hidden_layers = 3,
            noise_timesteps = 100, **kwargs):

        assert next_layer_size_multiplier > 1.0, "next_layer_size_multiplier must be > 1.0"
        assert encoding_size <= self.n_items, "encoding_size must be <= the number of items"

        encoder_architecture = generate_autoencoder_architecture(encoding_size, self.n_items, next_layer_size_multiplier, max_parameters, max_n_hidden_layers)[::-1]

        self._print("Architecture: {}".format(encoder_architecture))

        super().fit(encoder_architecture=encoder_architecture, noise_timesteps = int(noise_timesteps),
                    **kwargs)



    def _init_model(self):

        positional_encoding = SinusoidalPositionalEncoding(embedding_size = self.n_items, device=self.device)
        noise_schedule = LinearNoiseSchedule(start_beta = self.start_beta, end_beta = self.end_beta,
                                             device=self.device, noise_timesteps = self.noise_timesteps)

        denoiser_model = SNoDiagonalDenseAndAutoencoderModel(encoder_architecture = self.encoder_architecture,
                                          activation_function = _get_activation_function(self.activation_function),
                                          use_batch_norm = self.use_batch_norm,
                                          use_dropout = self.use_dropout,
                                          dropout_p = self.dropout_p,
                                          device=self.device)

        self._model = _GaussianDiffusionModel(denoiser_model,
                                              noise_schedule,
                                              positional_encoding,
                                              noise_timesteps = self.noise_timesteps,
                                              # inference_timesteps = self.inference_timesteps,
                                              # loss_type = 'l1',
                                              objective = self.objective,
                                              ).to(self.device)




class DiffusionSelfAttentionRecommender(DiffusionRecommender):
    """
    Diffusion model based on the user profile, which uses an autoencoder with residual connections as denoising model
    # TODO this was never finished and probably does not work

    This is the class that will be used when running an optimization as some hyperparameters are provided in a different way,
    in particular, the autoencoder architecture is parametrized using the encoding size and the layer multiplier.

    Example, if encoding = 50, next layer multiplier = 3, max hidden layers = 2
    the architecture is:
    [n_items, 450, 150, 50, 150, 450, n_items]
    """

    RECOMMENDER_NAME = "DiffusionSelfAttentionRecommender"

    def fit(self, encoding_size = 50, next_layer_size_multiplier = 2, max_parameters = np.inf, max_n_hidden_layers = 3,
            noise_timesteps = 100, **kwargs):

        assert next_layer_size_multiplier > 1.0, "next_layer_size_multiplier must be > 1.0"
        assert encoding_size <= self.n_items, "encoding_size must be <= the number of items"

        encoder_architecture = generate_autoencoder_architecture(encoding_size, self.n_items, next_layer_size_multiplier, max_parameters, max_n_hidden_layers)[::-1]

        self._print("Architecture: {}".format(encoder_architecture))

        super().fit(encoder_architecture=encoder_architecture, noise_timesteps = int(noise_timesteps),
                    **kwargs)



    def _init_model(self):

        positional_encoding = SinusoidalPositionalEncoding(embedding_size = self.n_items, device=self.device)
        noise_schedule = LinearNoiseSchedule(start_beta = self.start_beta, end_beta = self.end_beta,
                                             device=self.device, noise_timesteps = self.noise_timesteps)

        denoiser_model = MultiHeadAttentionBlock()

        self._model = _GaussianDiffusionModel(denoiser_model,
                                              noise_schedule,
                                              positional_encoding,
                                              noise_timesteps = self.noise_timesteps,
                                              # inference_timesteps = self.inference_timesteps,
                                              # loss_type = 'l1',
                                              objective = self.objective,
                                              ).to(self.device)






class DiffusionResidualAutoencoderRecommender_OptimizerMask(DiffusionRecommender):
    """
    Diffusion model based on the user profile, which uses an autoencoder with residual connections as denoising model
    # TODO this was never finished and probably does not work

    This is the class that will be used when running an optimization as some hyperparameters are provided in a different way,
    in particular, the autoencoder architecture is parametrized using the encoding size and the layer multiplier.

    Example, if encoding = 50, next layer multiplier = 3, max hidden layers = 2
    the architecture is:
    [n_items, 450, 150, 50, 150, 450, n_items]
    """

    RECOMMENDER_NAME = "DiffusionResidualAutoencoderRecommender"

    def fit(self, encoding_size = 50, next_layer_size_multiplier = 2, max_parameters = np.inf, max_n_hidden_layers = 3,
            noise_timesteps = 100, **kwargs):

        assert next_layer_size_multiplier > 1.0, "next_layer_size_multiplier must be > 1.0"
        assert encoding_size <= self.n_items, "encoding_size must be <= the number of items"

        encoder_architecture = generate_autoencoder_architecture(encoding_size, self.n_items, next_layer_size_multiplier, max_parameters, max_n_hidden_layers)[::-1]

        self._print("Architecture: {}".format(encoder_architecture))

        super().fit(encoder_architecture=encoder_architecture, noise_timesteps = int(noise_timesteps),
                    **kwargs)



    def _init_model(self):

        positional_encoding = SinusoidalPositionalEncoding(embedding_size = self.n_items, device=self.device)
        noise_schedule = LinearNoiseSchedule(start_beta = self.start_beta, end_beta = self.end_beta,
                                             device=self.device, noise_timesteps = self.noise_timesteps)

        denoiser_model = ResidualAutoencoderModel(encoder_architecture = self.encoder_architecture,
                                          activation_function = _get_activation_function(self.activation_function),
                                          use_batch_norm = self.use_batch_norm,
                                          use_dropout = self.use_dropout,
                                          dropout_p = self.dropout_p,
                                          device=self.device)

        self._model = _GaussianDiffusionModel(denoiser_model,
                                              noise_schedule,
                                              positional_encoding,
                                              noise_timesteps = self.noise_timesteps,
                                              # inference_timesteps = self.inference_timesteps,
                                              # loss_type = 'l1',
                                              objective = self.objective,
                                              ).to(self.device)



class DiffusionVariationalRecommender_OptimizerMask(DiffusionRecommender):
    """
    Diffusion model based on the user profile, which uses a variational autoencoder as denoising model
    # TODO this was never finished and probably does not work

    This is the class that will be used when running an optimization as some hyperparameters are provided in a different way,
    in particular, the autoencoder architecture is parametrized using the encoding size and the layer multiplier.

    Example, if encoding = 50, next layer multiplier = 3, max hidden layers = 2
    the architecture is:
    [n_items, 450, 150, 50 (+50), 150, 450, n_items]
    """

    RECOMMENDER_NAME = "DiffusionVariationalRecommender"

    def fit(self, encoding_size = 50, next_layer_size_multiplier = 2, max_parameters = np.inf, max_n_hidden_layers = 3,
            noise_timesteps = 100, **kwargs):

        assert next_layer_size_multiplier > 1.0, "next_layer_size_multiplier must be > 1.0"
        assert encoding_size <= self.n_items, "encoding_size must be <= the number of items"

        encoder_architecture = generate_autoencoder_architecture(encoding_size, self.n_items, next_layer_size_multiplier, max_parameters, max_n_hidden_layers)[::-1]

        self._print("Architecture: {}".format(encoder_architecture))

        super().fit(encoder_architecture=encoder_architecture, noise_timesteps = int(noise_timesteps),
                    **kwargs)



    def _init_model(self):

        positional_encoding = SinusoidalPositionalEncoding(embedding_size = self.n_items, device=self.device)
        noise_schedule = LinearNoiseSchedule(start_beta = self.start_beta, end_beta = self.end_beta,
                                             device=self.device, noise_timesteps = self.noise_timesteps)

        denoiser_model = VariationalAutoencoderModel(encoder_architecture = self.encoder_architecture,
                                          activation_function = _get_activation_function(self.activation_function),
                                          use_batch_norm = self.use_batch_norm,
                                          use_dropout = self.use_dropout,
                                          dropout_p = self.dropout_p,
                                          device=self.device)

        self._model = _GaussianDiffusionModel(denoiser_model,
                                              noise_schedule,
                                              positional_encoding,
                                              noise_timesteps = self.noise_timesteps,
                                              # inference_timesteps = self.inference_timesteps,
                                              # loss_type = 'l1',
                                              objective = self.objective,
                                              ).to(self.device)




class DiffusionTowerRecommender_OptimizerMask(DiffusionRecommender):
    """
    Diffusion model based on the user profile, which uses a tower network as denoising model

    This is the class that will be used when running an optimization as some hyperparameters are provided in a different way,
    in particular, the architecture is parametrized using the encoding size and the number of layers.

    Example, if encoding = 50, n_layers = 3,
    the architecture is:
    [n_items, 50, 50, 50, n_items]
    """

    RECOMMENDER_NAME = "DiffusionTowerRecommender"

    def fit(self, encoding_size = 50, n_layers = 2, max_parameters = np.inf, max_n_hidden_layers = 3,
            noise_timesteps = 100, **kwargs):

        assert encoding_size <= self.n_items, "encoding_size must be <= the number of items"

        tower_architecture = generate_tower_architecture(self.n_items, self.n_items, encoding_size, max_parameters, max_n_hidden_layers)

        self._print("Architecture: {}".format(tower_architecture))

        super().fit(encoder_architecture=tower_architecture, noise_timesteps = int(noise_timesteps),
                    **kwargs)


    def _init_model(self):

        positional_encoding = SinusoidalPositionalEncoding(embedding_size = self.n_items, device=self.device)
        noise_schedule = LinearNoiseSchedule(start_beta = self.start_beta, end_beta = self.end_beta,
                                             device=self.device, noise_timesteps = self.noise_timesteps)

        denoiser_model = TowerModel(tower_architecture = self.encoder_architecture,
                                    activation_function = _get_activation_function(self.activation_function),
                                    use_batch_norm = self.use_batch_norm,
                                    use_dropout = self.use_dropout,
                                    dropout_p = self.dropout_p,
                                    device=self.device)

        self._model = _GaussianDiffusionModel(denoiser_model,
                                              noise_schedule,
                                              positional_encoding,
                                              noise_timesteps = self.noise_timesteps,
                                              # inference_timesteps = self.inference_timesteps,
                                              # loss_type = 'l1',
                                              objective = self.objective,
                                              ).to(self.device)


class DiffusionSDenseRecommender_OptimizerMask(DiffusionRecommender):
    """
    Diffusion model based on the user profile, which uses a dense item-item similarity (with the diagonal) as denoising model
    R_tilde = R S
    """

    RECOMMENDER_NAME = "DiffusionSDenseRecommender"

    def fit(self, encoding_size = None, #TODO remove
            max_parameters = np.inf,
            noise_timesteps = 100, **kwargs):

        super().fit(noise_timesteps = int(noise_timesteps), **kwargs)


    def _init_model(self):

        positional_encoding = SinusoidalPositionalEncoding(embedding_size = self.n_items, device=self.device)
        noise_schedule = LinearNoiseSchedule(start_beta = self.start_beta, end_beta = self.end_beta,
                                             device=self.device, noise_timesteps = self.noise_timesteps)

        denoiser_model = SDenseModel(n_items = self.n_items,
                                     device=self.device)

        self._model = _GaussianDiffusionModel(denoiser_model,
                                              noise_schedule,
                                              positional_encoding,
                                              noise_timesteps = self.noise_timesteps,
                                              # inference_timesteps = self.inference_timesteps,
                                              # loss_type = 'l1',
                                              objective = self.objective,
                                              ).to(self.device)


class DiffusionSNoDiagonalDenseRecommender_OptimizerMask(DiffusionRecommender):
    """
    Diffusion model based on the user profile, which uses a dense item-item similarity (without the diagonal) as denoising model
    R_tilde = R S
    """

    RECOMMENDER_NAME = "DiffusionSNoDiagonalDenseRecommender"

    def fit(self, encoding_size = 50, max_parameters = np.inf,
            noise_timesteps = 100, **kwargs):

        assert encoding_size <= self.n_items, "encoding_size must be <= the number of items"
        self.encoding_size = encoding_size
        super().fit(noise_timesteps = int(noise_timesteps), **kwargs)


    def _init_model(self):

        positional_encoding = SinusoidalPositionalEncoding(embedding_size = self.n_items, device=self.device)
        noise_schedule = LinearNoiseSchedule(start_beta = self.start_beta, end_beta = self.end_beta,
                                             device=self.device, noise_timesteps = self.noise_timesteps)

        denoiser_model = SNoDiagonalDenseModel(n_items = self.n_items,
                                     device=self.device)

        self._model = _GaussianDiffusionModel(denoiser_model,
                                              noise_schedule,
                                              positional_encoding,
                                              noise_timesteps = self.noise_timesteps,
                                              # inference_timesteps = self.inference_timesteps,
                                              # loss_type = 'l1',
                                              objective = self.objective,
                                              ).to(self.device)


class DiffusionSFactorizedRecommender_OptimizerMask(DiffusionRecommender):
    """
    Diffusion model based on the user profile, which uses two identical embedding as denoising model
    R_tilde = R VV.t
    """

    RECOMMENDER_NAME = "DiffusionSFactorizedRecommender"

    def fit(self, encoding_size = 50, max_parameters = np.inf,
            noise_timesteps = 100, **kwargs):

        assert encoding_size <= self.n_items, "encoding_size must be <= the number of items"
        self.encoding_size = encoding_size
        super().fit(noise_timesteps = int(noise_timesteps), **kwargs)


    def _init_model(self):

        positional_encoding = SinusoidalPositionalEncoding(embedding_size = self.n_items, device=self.device)
        noise_schedule = LinearNoiseSchedule(start_beta = self.start_beta, end_beta = self.end_beta,
                                             device=self.device, noise_timesteps = self.noise_timesteps)

        denoiser_model = SFactorizedModel(embedding_size = self.encoding_size,
                                          n_items = self.n_items,
                                          device=self.device)

        self._model = _GaussianDiffusionModel(denoiser_model,
                                              noise_schedule,
                                              positional_encoding,
                                              noise_timesteps = self.noise_timesteps,
                                              # inference_timesteps = self.inference_timesteps,
                                              # loss_type = 'l1',
                                              objective = self.objective,
                                              ).to(self.device)



class DiffusionAsySVDRecommender_OptimizerMask(DiffusionRecommender):
    """
    Diffusion model based on the user profile, which uses asymmetric SVD as denoising model
    R_tilde = RXY
    """

    RECOMMENDER_NAME = "DiffusionAsySVDRecommender"

    def fit(self, encoding_size = 50, max_parameters = np.inf,
            noise_timesteps = 100, **kwargs):

        assert encoding_size <= self.n_items, "encoding_size must be <= the number of items"
        self.encoding_size = encoding_size
        super().fit(noise_timesteps = int(noise_timesteps), **kwargs)


    def _init_model(self):

        positional_encoding = SinusoidalPositionalEncoding(embedding_size = self.n_items, device=self.device)
        noise_schedule = LinearNoiseSchedule(start_beta = self.start_beta, end_beta = self.end_beta,
                                             device=self.device, noise_timesteps = self.noise_timesteps)

        denoiser_model = AsySVDModel(embedding_size = self.encoding_size,
                                      n_items = self.n_items,
                                      device=self.device)

        self._model = _GaussianDiffusionModel(denoiser_model,
                                              noise_schedule,
                                              positional_encoding,
                                              noise_timesteps = self.noise_timesteps,
                                              # inference_timesteps = self.inference_timesteps,
                                              # loss_type = 'l1',
                                              objective = self.objective,
                                              ).to(self.device)




class DiffusionItemSVDRecommender_OptimizerMask(DiffusionRecommender):
    """
    Diffusion model based on the user profile, which uses the SVD decomposition as denoising model
    R_tilde = V Sigma V.t
    The model initializes V on the original R and then learns the Sigma values
    """

    RECOMMENDER_NAME = "DiffusionItemSVDRecommender"

    def fit(self, encoding_size = 50, max_parameters = np.inf,
            noise_timesteps = 100, **kwargs):

        assert encoding_size <= self.n_items, "encoding_size must be <= the number of items"
        self.encoding_size = encoding_size
        super().fit(noise_timesteps = int(noise_timesteps), **kwargs)


    def _init_model(self):

        positional_encoding = SinusoidalPositionalEncoding(embedding_size = self.n_items, device=self.device)
        noise_schedule = LinearNoiseSchedule(start_beta = self.start_beta, end_beta = self.end_beta,
                                             device=self.device, noise_timesteps = self.noise_timesteps)

        denoiser_model = ItemSVDModel(embedding_size = self.encoding_size,
                                      URM_train = self.URM_train,
                                      device=self.device)

        self._model = _GaussianDiffusionModel(denoiser_model,
                                              noise_schedule,
                                              positional_encoding,
                                              noise_timesteps = self.noise_timesteps,
                                              # inference_timesteps = self.inference_timesteps,
                                              # loss_type = 'l1',
                                              objective = self.objective,
                                              ).to(self.device)
