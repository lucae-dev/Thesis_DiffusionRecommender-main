from Diffusion.AttentionModels import SimpleAttentionDiffusionModel
from Diffusion.DiffusionRecommender import _GaussianDiffusionModel, SimpleAutoencoder, _get_optimizer
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.DataIO import DataIO

import os, shutil, copy, math
import torch.nn.functional as F
from tqdm.auto import tqdm

from Diffusion.DenoisingArchitectures import *
from Diffusion.PositionalEncoding import SinusoidalPositionalEncoding
from Diffusion.NoiseSchedule import LinearNoiseSchedule

class MultiBlockAttentionDiffusionRecommender(BaseRecommender, Incremental_Training_Early_Stopping):
    """
    Diffusion model based on user profiles, using a single self-attention layer as denoising architecture
    """

    RECOMMENDER_NAME = "MultiBlockAttentionDiffusionRecommender"

    def __init__(self, URM_train, use_gpu = True, verbose = True):
        super(MultiBlockAttentionDiffusionRecommender, self).__init__(URM_train, verbose = verbose)

        if use_gpu:
            # Check for CUDA availability (NVIDIA GPUs)
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                torch.cuda.empty_cache()
            # Check for MPS availability (Apple Silicon GPUs)
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
        
    def get_diffusion_parameters(self):
        return self.diffusion_model.parameters()

    def _compute_item_score(self, user_id_array, items_to_compute = None):
        """
    
        :param user_id_array:
        :param items_to_compute:
        :return:
        """            

        user_batch_tensor = self.URM_train[user_id_array]
        # Convert CSR matrix to a dense numpy array directly
        user_batch_dense_np = user_batch_tensor.toarray()

        # Convert the dense numpy array to a PyTorch tensor
        # and move it to the appropriate device
        if str(self.device) == 'mps':
            user_batch_tensor = torch.tensor(user_batch_dense_np, dtype=torch.float32, device='cpu').to('mps')
        else:
        # Transferring only the sparse structure to reduce the data transfer
            user_batch_tensor = torch.sparse_csr_tensor(user_batch_tensor.indptr,
                                                        user_batch_tensor.indices,
                                                        user_batch_tensor.data,
                                                        size=user_batch_tensor.shape,
                                                        dtype=torch.float32,
                                                        device=self.device,
                                                        requires_grad=False).to_dense()

          
        user_profile_inference = self.diffusion_model.inference(user_batch_tensor, self.inference_timesteps)
        
        if items_to_compute is None:
            item_scores = user_profile_inference
        else:
            item_scores = - np.ones((len(user_id_array), self.n_items), dtype=np.float32)*np.inf

        return item_scores



    def _init_model(self):

        positional_encoding = SinusoidalPositionalEncoding(embedding_size = self.n_items, device=self.device)
        noise_schedule = LinearNoiseSchedule(start_beta = self.start_beta, end_beta = self.end_beta,
                                             device=self.device, noise_timesteps = self.noise_timesteps)

        denoiser_model = MultiBlockEncoder(d_model = self.embeddings_dim, d_ff = self.d_ff, h = self.heads, device = self.device, dropout=self.dropout_p, n_blocks = self.attention_blocks).to(self.device)

        gaussian_model = _GaussianDiffusionModel(denoiser_model,
                                              noise_schedule,
                                              positional_encoding,
                                              noise_timesteps = self.noise_timesteps,
                                              # inference_timesteps = self.inference_timesteps,
                                              # loss_type = 'l1',
                                              objective = self.objective,
                                              ).to(self.device)
        
        encoder_model = SimpleAutoencoder(self.n_items, self.embeddings_dim, device=self.device)

        self.diffusion_model = SimpleAttentionDiffusionModel(gaussian_model = gaussian_model,
                                                             encoder_model = encoder_model,
                                                             denoiser_model = denoiser_model).to(self.device)

    def fit(self, epochs = 300,
            batch_size = 128,
            embeddings_dim = 128,
            heads = 1,
            attention_blocks = 1,
            d_ff = 512,
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
        self.attention_blocks = attention_blocks
        self.d_ff = d_ff
        self._init_model()

        self._optimizer = _get_optimizer(self.sgd_mode.lower(), self.diffusion_model, self.learning_rate, self.l2_reg)

        self._update_best_model()

        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        self.loss_list = []

        self._train_with_early_stopping(epochs,
                                        algorithm_name = self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))
        # prof.export_chrome_trace("trace.json")

        self._print("Training complete")

        self.diffusion_model.load_state_dict(self._model_best)

        # Set model in evaluation mode (disables dropout, for example)
        self.diffusion_model.eval()


    def _prepare_model_for_validation(self):
        self.diffusion_model.eval() # Particularly important if we use dropout or batch norm layers


    def _update_best_model(self):
        self._model_best = copy.deepcopy(self.diffusion_model.state_dict())

    def forward(self, x_start_batch, t):
        """

        :param x_start:
        :param t:
        :return:
        """
        return self.diffusion_model.forward(x_start_batch, t)

    def _run_epoch(self, num_epoch):

        self.current_epoch_training_loss = 0
        self.diffusion_model.train()

        #batches must become number of batches, the batch size will another parameter -> add parameter n_batches
        num_batches_per_epoch = math.ceil(len(self.warm_user_ids) / self.batch_size)

        iterator = tqdm(range(num_batches_per_epoch)) if self.show_progress_bar else range(num_batches_per_epoch)

        for _ in iterator:

            user_batch = torch.LongTensor(np.random.choice(self.warm_user_ids, size=self.batch_size))

            if (str(self.device) == 'mps'):
                sparse_device = 'cpu'
            else:
                sparse_device = self.device

            user_batch_tensor = self.URM_train[user_batch]
            # Convert CSR matrix to a dense numpy array directly
            user_batch_dense_np = user_batch_tensor.toarray()

            # Convert the dense numpy array to a PyTorch tensor
            # and move it to the appropriate device
            if str(self.device) == 'mps':
                user_batch_tensor = torch.tensor(user_batch_dense_np, dtype=torch.float32, device='cpu').to('mps')
            else:
            # Transferring only the sparse structure to reduce the data transfer
                user_batch_tensor = torch.sparse_csr_tensor(user_batch_tensor.indptr,
                                                            user_batch_tensor.indices,
                                                            user_batch_tensor.data,
                                                            size=user_batch_tensor.shape,
                                                            dtype=torch.float32,
                                                            device=self.device,
                                                            requires_grad=False).to_dense()
            #if(str(self.device)=="mps"):
            #   user_batch_tensor = user_batch_tensor.to("mps")  #Reassign the tensor

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


        if (self.verbose == True):
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
            "encoder_architecture": self.encoder_model,
            "activation_function": self.activation_function,
            "start_beta": self.start_beta,
            "end_beta": self.end_beta,
            "noise_timesteps": self.noise_timesteps,
            "objective": self.objective,
            "embeddings_dim": self.embeddings_dim,
            "heads": self.heads,
            "inference_timesteps": self.inference_timesteps,
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



class MultiBlockAttentionDiffusionRecommenderInf(MultiBlockAttentionDiffusionRecommender):

    RECOMMENDER_NAME = "MultiBlockAttentionDiffusionRecommenderInf"

    def _compute_item_score(self, user_id_array, items_to_compute = None):
        n_batches = np.ceil(len(user_id_array) / self.batch_size).astype(int)
        user_profile_inference_temp = []
        for batch_num in range(n_batches):
            # Calculate start and end indices for the current batch
            start_idx = batch_num * self.batch_size
            end_idx = min((batch_num + 1) * self.batch_size, len(user_id_array))

            # Fetch the batch of user IDs
            batch_user_ids = user_id_array[start_idx:end_idx]

            # Convert CSR matrix to a dense numpy array for the current batch
            user_batch_tensor = self.URM_train[batch_user_ids]
            user_batch_dense_np = user_batch_tensor.toarray()

            # Convert the dense numpy array to a PyTorch tensor and move it to the appropriate device
            if str(self.device) == 'mps':
                user_batch_tensor = torch.tensor(user_batch_dense_np, dtype=torch.float32, device='cpu').to('mps')
            else:
                # Transferring only the sparse structure to reduce the data transfer
                user_batch_tensor = torch.sparse_csr_tensor(user_batch_tensor.indptr,
                                                            user_batch_tensor.indices,
                                                            user_batch_tensor.data,
                                                            size=user_batch_tensor.shape,
                                                            dtype=torch.float32,
                                                            device=self.device,
                                                            requires_grad=False).to_dense()

            # Perform inference with the diffusion model for the current batch
            user_profile_inference_temp.append(self.diffusion_model.inference(user_batch_tensor, self.inference_timesteps))

        
        user_profile_inference = np.vstack(user_profile_inference_temp)
        
        if items_to_compute is None:
            item_scores = user_profile_inference
        else:
            item_scores = - np.ones((len(user_id_array), self.n_items), dtype=np.float32)*np.inf

        return item_scores

