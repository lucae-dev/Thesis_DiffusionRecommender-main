#WSAD - Recommender

import math

import numpy as np
import torch
import tqdm

from Diffusion.AttentionModels import SimpleAttentionDiffusionModel
from Diffusion.DenoisingArchitectures import MultiBlockEncoder
from Diffusion.MultiBlockAttentionDiffusionRecommenderSimilarity import MultiBlockAttentionDiffusionRecommenderSimilarity
from Diffusion.DiffusionRecommender import _GaussianDiffusionModel, SimpleAutoencoder
from Diffusion.NoiseSchedule import LinearNoiseSchedule
from Diffusion.PositionalEncoding import SinusoidalPositionalEncoding
from Diffusion.similarity_models import TwoRandomWalksSampler

# SAD - R


class WSAD_Recommender(MultiBlockAttentionDiffusionRecommenderSimilarity):
    RECOMMENDER_NAME = "W-SAD"


    def _init_model(self):

        positional_encoding = SinusoidalPositionalEncoding(embedding_size = self.n_items, device=self.device)
        noise_schedule = LinearNoiseSchedule(start_beta = self.start_beta, end_beta = self.end_beta,
                                            device=self.device, noise_timesteps = self.noise_timesteps)

        denoiser_model = MultiBlockEncoder(d_model = self.embeddings_dim, d_ff = self.d_ff, h = self.heads, device = self.device, dropout=self.dropout_p, n_blocks = self.attention_blocks, seq_len = self.batch_size, similarity_weight=self.similarity_weight).to(self.device)

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
        
        self.sampler = TwoRandomWalksSampler(self.URM_train, self.warm_user_ids) 

    def _run_epoch(self, num_epoch):

        self.current_epoch_training_loss = 0
        self.diffusion_model.train()

        #batches must become number of batches, the batch size will another parameter -> add parameter n_batches
        num_batches_per_epoch = math.ceil(len(self.warm_user_ids) / self.batch_size)

        iterator = tqdm(range(num_batches_per_epoch)) if self.show_progress_bar else range(num_batches_per_epoch)

        for _ in iterator:

            user_batch = torch.LongTensor(np.random.choice(self.warm_user_ids, size=self.batch_size))

            # build sub similarity matrix between user_batch users only
            similarity_matrix = self.sampler.similarity_matrix[user_batch][:, user_batch]

            user_batch_tensor = self.URM_train[user_batch]
            # Convert CSR matrix to a dense numpy array directly
            user_batch_dense_np = user_batch_tensor.toarray()

            # Convert the dense numpy array to a PyTorch tensor
            # and move it to the appropriate device
            if str(self.device) == 'mps':
                user_batch_tensor = torch.tensor(user_batch_dense_np, dtype=torch.float32, device='mps')
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
            loss = self.forward(user_batch_tensor, t, similarity_matrix)

            # Compute gradients given current loss
            loss.backward()
            # torch.nn.utils.clip_grad_norm(self._model.parameters(), max_norm=1.0)

            # Apply gradient using the selected optimizer
            self._optimizer.step()

            self.current_epoch_training_loss += loss.item()


        if (self.verbose == True):
         self._print("Epoch {}, loss {:.2E}".format(num_epoch, self.current_epoch_training_loss))
        # self.loss_list.append(self.current_epoch_training_loss)




    def _compute_item_score(self, user_id_array, items_to_compute = None):
       
        n_batches = np.ceil(len(user_id_array) / self.batch_size).astype(int)
        user_profile_inference_temp = []

        for batch_num in range(n_batches):
            # Calculate start and end indices for the current batch
            start_idx = batch_num * self.batch_size
            end_idx = min((batch_num + 1) * self.batch_size, len(user_id_array))

            # Fetch the batch of user IDs
            batch_user_ids = user_id_array[start_idx:end_idx]

            similarity_matrix = self.sampler.similarity_matrix[batch_user_ids][:, batch_user_ids]

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
            user_profile_inference_temp.append(self.diffusion_model.inference(user_batch_tensor, self.inference_timesteps, similarity_matrix = similarity_matrix))

        
        user_profile_inference = np.vstack(user_profile_inference_temp)
        
        if items_to_compute is None:
            item_scores = user_profile_inference
        else:
            item_scores = - np.ones((len(user_id_array), self.n_items), dtype=np.float32)*np.inf

        return item_scores