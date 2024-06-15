from torch import nn
import torch
import torch.nn.functional as F

class SimpleAttentionDiffusionModel(nn.Module):
    def __init__(self, gaussian_model, encoder_model, denoiser_model):
        super().__init__()
        self._model = gaussian_model
        self.encoder_model = encoder_model
        self.denoiser_model = denoiser_model

    def forward(self, x_start_batch, t, similarity_matrix = None):
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
        
        denoiser_prediction = self.denoiser_model.forward(x_noisy, similarity_matrix, None)
        denosier_loss = self.denoiser_model.loss()

        if self._model.objective == 'pred_noise':
            target = gaussian_noise
        elif self._model.objective == 'pred_x0':
            target = x_emb_batch
        else:
            raise ValueError(f'unknown objective {self._model.objective}')

        decoded_prediction = self.encoder_model.decode(denoiser_prediction)

        loss_batch = F.mse_loss(decoded_prediction, x_start_batch) + F.mse_loss(denoiser_prediction, target) + denosier_loss

        return loss_batch
    
    def inference(self, user_profile_batch, inference_timesteps, similarity_matrix = None):
        for inference_timestep in range(inference_timesteps, 0, -1):
            x_emb_batch = self.encoder_model.encode(user_profile_batch)
            
            user_profile_inference_emb = self._model.sample_from_user_profile(x_emb_batch, inference_timestep, similarity_matrix)
            
            user_profile_inference = self.encoder_model.decode(user_profile_inference_emb).cpu().detach().numpy()
        return user_profile_inference

class MultiBlockAttentionDiffusionModel(nn.Module):
    def __init__(self, gaussian_model, encoder_model, denoiser_model):
        super().__init__()
        self._model = gaussian_model
        self.encoder_model = encoder_model
        self.denoiser_model = denoiser_model

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

        if self._model.objective == 'pred_noise':
            target = gaussian_noise
        elif self._model.objective == 'pred_x0':
            target = x_emb_batch
        else:
            raise ValueError(f'unknown objective {self._model.objective}')

        decoded_prediction = self.encoder_model.decode(denoiser_prediction)

        loss_batch = F.mse_loss(decoded_prediction, x_start_batch) + F.mse_loss(denoiser_prediction, target) + denosier_loss

        return loss_batch
    
    def inference(self, user_profile_batch, inference_timesteps):
        x_emb_batch = self.encoder_model.encode(user_profile_batch)
        
        user_profile_inference_emb = self._model.sample_from_user_profile(x_emb_batch, inference_timesteps)
        
        user_profile_inference = self.encoder_model.decode(user_profile_inference_emb).cpu().detach().numpy()
        return user_profile_inference
