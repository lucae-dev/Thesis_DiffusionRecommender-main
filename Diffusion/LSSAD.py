
import numpy as np
import torch
from tqdm import tqdm
from Diffusion.MultiBlockAttentionDiffusionRecommenderSimilarity import MultiBlockAttentionDiffusionRecommenderSimilarity


class LSSAD(MultiBlockAttentionDiffusionRecommenderSimilarity):
    RECOMMENDER_NAME = "LS_SAD"
    
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
        
        self.batches = self.sampler.batch_users(batch_size)

        super().fit(epochs,
            batch_size ,
            embeddings_dim,
            heads ,
            attention_blocks,
            d_ff ,
            l2_reg ,
            sgd_mode ,
            learning_rate,
            noise_timesteps,
            inference_timesteps,
            show_progress_bar,
            start_beta,
            end_beta,
            activation_function,
            use_batch_norm,
            use_dropout,
            dropout_p,
            objective,
            **earlystopping_kwargs)
                
    def _run_epoch(self, num_epoch):

        self.current_epoch_training_loss = 0
        self.diffusion_model.train()

        #batches must become number of batches, the batch size will another parameter -> add parameter n_batches
        # num_batches_per_epoch = math.ceil(len(self.warm_user_ids) / self.batch_size)

        #len_warm_ids = len(self.warm_user_ids)

        #iterator = tqdm(range(num_batches_per_epoch)) if self.show_progress_bar else range(num_batches_per_epoch)
        #iterator = tqdm(range(len_warm_ids)) if self.show_progress_bar else range(len_warm_ids)

        for user_batch in self.batches:

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



    def _compute_item_score(self, user_id_array, items_to_compute = None):
        """

        :param user_id_array:
        :param items_to_compute:
        :return:
        """          
        user_profile_inference_temp = []

        user_batches_indxs = self.find_user_batches(self.batches, user_id_array)
    
        #print(user_batches_indxs)

        # Use a set to ensure unique batch indices
        unique_user_batches = list(set(user_batches_indxs))
        print(len(self.batches))
        print(unique_user_batches)
        # Iterate through the unique batches
        for batch_index in unique_user_batches:
            user_batch = self.batches[batch_index]
            #print(user_batch)
            # Find indices of user IDs in user_batch that are also present in user_ids
            users_idx_to_append = [idx for (idx, user_id) in enumerate(user_batch) if user_id in user_id_array]

            user_batch_tensor = self.URM_train[user_batch]
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


            inference_results = self.diffusion_model.inference(user_batch_tensor, self.inference_timesteps)

            for idx in users_idx_to_append:
                user_profile_inference_temp.append(inference_results[idx])
        
        user_profile_inference = np.vstack(user_profile_inference_temp)
    
        if items_to_compute is None:
            item_scores = user_profile_inference
        else:
            item_scores = - np.ones((len(user_id_array), self.n_items), dtype=np.float32)*np.inf

        return item_scores
    

    def find_user_batches(self, batches, user_ids):
        # Create a dictionary to map user ID to batch index
        user_to_batch = {}

        # Iterate through each batch and user ID to populate the dictionary
        for batch_index, batch in enumerate(batches):
            for user_id in batch:
                user_to_batch[user_id] = batch_index

        # Find the batch index for each user ID in the user_ids array
        user_batches = [user_to_batch[user_id] for user_id in user_ids]

        return user_batches