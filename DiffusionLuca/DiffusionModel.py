#Gaussian diffusion model rewritten by me
import math
import numpy as np
from torch import nn
import torch 
import torch.nn.functional as F
import scipy.sparse as sps
from Diffusion.DenoisingArchitectures import MultiHeadAttentionBlock
from Diffusion.NoiseSchedule import LinearNoiseSchedule
from Diffusion.PositionalEncoding import SinusoidalPositionalEncoding

from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping

class _GaussianDiffusionModel(nn.Module):
    def __init__(self,
                 denoiser_model,
                 noise_schedule,
                 positional_encoding,
                 noise_timesteps = 1000,
                 objective = 'pred_noise'
                 ):
    
        super().__init__()

        self.denoiser_model= denoiser_model
        self.noise_schedule = noise_schedule
        self.positional_encoding = positional_encoding
        self.noise_timesteps = noise_timesteps
        
        assert objective in {'pred_noise', 'pred_x0'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start)'
        self.objective = objective

    #add noise to sample, eta is the gaussian noise (noise at t = N(sqrt(alpha)*x_start, 1-alpha*eta), alpha at t alpha_t = 1-beta_t a_signed = cumprod(alpha_t))
    def q_sample(self, x0, t, eta = None):
        a_signed = self.noise_schedule.get_a_signed(t)
        batch_size = len(t)
        x_noisy = torch.sqrt(a_signed).reshape(batch_size,1)*x0 + torch.sqrt(1-a_signed).reshape(batch_size, 1) * eta
        
        return x_noisy

    def q_batch_sample(self, x0_batch, t, eta = None):
        a_signed = self.noise_schedule.get_a_signed(t)
        batch_size = len(t)
        x0_noisy = torch.sqrt(a_signed).reshape(batch_size,1) * x0_batch + torch.sqrt(1-a_signed).reshape(batch_size, 1) * eta
        
        return x0_noisy

    #Add noise to the input
    def forward(self, x0_batch, t, eta = None):
        eta = torch.randn_like(x0_batch) #eta is the sampled gaussian noise
        
        #noise sample
        x_noisy = self.q_sample(x0 = x0_batch, t = t, eta = eta)
        
        #add timestep information to noise
        x_noisy = x_noisy + self.positional_encoding.get_encoding(t)

        #now i want to teach the denoiser to learn to predict the gaussian noise eta, so the result of this prediction ideally is equal to eta. (I can also teach it to predict x0 directly)
        denoiser_prediction = self.denoiser_model(x_noisy)
        #probably loss between input (x_noisy) and output (prediction)
        denoiser_loss = self.denoiser_model.loss() #isn't this all 0???????? or does it get automatically updated in the forward process for a characteristic of the the nn somehting??? Yes, it most definitly is

        if self.objective == 'pred_noise':
            target = eta
        elif self.objective == 'pred_x0':
            target = x0_batch
        else:
            raise ValueError(f'Diffusion (Gaussian) model: unknown objective{self.objective}')

        #PERCHE AGGIUNGO LA DENOISER LOSS(DIFFERENZA TRA X_NOISY E PREDICTION, DOVE PREDICTION IO VOGLIO SIA ETA O X0, QUIDNI LA LOSS NON LA VOGLIO PICCOLA(SINGIFICHEREBBE RICOSTRUIRE X_NOISY))
        loss_batch = F.mse_loss(denoiser_prediction, target) + denoiser_loss
        return loss_batch

    #inference for a user profile
    def sample_from_user_profile(self, user_profile, inference_timesteps):
        with torch.no_grad():
            #sample the gaussian noise eta (initialize to random from a normal distribuiton) to add noise to the user profile (since it is sampled randomly from a gaussian distribution it will be a different one from the eta we used up)
            eta = torch.randn_like(user_profile, device = user_profile.device)

            #add noise to the user profile at the timestep we passed
            user_profile_inference = self.q_sample(x0 = user_profile, t = torch.tensor([inference_timesteps],device = user_profile.device), eta = eta)

            for timestep in range(inference_timesteps, 0, -1):
                #add the positional information to decode
                user_profile_inference = user_profile_inference + self.positional_encoding(torch.tensor[timestep], device = user_profile.device)
                #CALOLO LA PREDICTION
                user_profile_inference = self.denoiser_model.forward(user_profile_inference)
                posterior_mean = user_profile * self.noise_schedule._posterior_mean_c_x_start[timestep] + user_profile_inference * self.noise_schedule._posterior_mean_c_x_t[timestep]
                if timestep > 0:
                    noise = torch.randn_like(user_profile, device = user_profile.device)
                    user_profile_inference = posterior_mean + (0.5*self.noise_schedule._posterior_log_variance_clipped[timestep]).exp() * noise
                else:
                    user_profile_inference = posterior_mean

        return user_profile_inference
    
    #inference for a user profile
    def sample_batch_from_user_profile(self, batch_profiles, inference_timesteps):
        with torch.no_grad():
            #sample the gaussian noise eta (initialize to random from a normal distribuiton) to add noise to the user profile (since it is sampled randomly from a gaussian distribution it will be a different one from the eta we used up)
            eta = torch.randn_like(batch_profiles, device = batch_profiles.device)

            #add noise to the user profile at the timestep we passed
            user_profile_inference = self.q_batch_sample(x0 = batch_profiles, t = torch.tensor([inference_timesteps], device = batch_profiles.device), eta = eta)

            for timestep in range(inference_timesteps, 0, -1):
                #add the positional information to decode
                user_profile_inference = user_profile_inference + self.positional_encoding(torch.tensor[timestep], device = batch_profiles.device)
                #CALOLO LA PREDICTION
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
            assert torch.cuda.is_available(), "GPU is requested but not available"
            self.device = torch.device("cuda:0")
            torch.cuda.empty_cache()
        else:
            self.device = torch.device("cpu:0")

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

        denoiser_model = MultiHeadAttentionBlock(d_model = 512, h = self.heads, device = self.device)

        self._model = _GaussianDiffusionModel(denoiser_model,
                                              noise_schedule,
                                              positional_encoding,
                                              noise_timesteps = self.noise_timesteps,
                                              # inference_timesteps = self.inference_timesteps,
                                              # loss_type = 'l1',
                                              objective = self.objective,
                                              ).to(self.device)


    def fit(self, epochs = 300,
            batch_size = 128,
            heads = 4,
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
        self.heads = heads

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

            #here you coudl transform the user profile to reduce dimensionality before passing tbem to the forward, the forward will learn this way to predict the reduced-size-user-profile (which is like an embedding), not the normal user profile (or the noise applied to embeddign)
            # you can do it just passing the user profiles trough a linear layer, just take the encoder of an autoencoder and get it trough it

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
