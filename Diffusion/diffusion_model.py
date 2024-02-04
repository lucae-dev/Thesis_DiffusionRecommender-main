#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 20/08/2022

@author: Maurizio Ferrari Dacrema
"""

# pip install denoising_diffusion_pytorch
# https://github.com/lucidrains/denoising-diffusion-pytorch

import torch
import numpy as np
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import tensorflow as tf

if __name__ == '__main__':

    # The dim parameter specifies the number of feature maps before the first down-sampling,
    # and the dim_mults parameter provides multiplicands for this value and successive down-samplings:

    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8)
    )

    # We pass in the U-Net model that we just defined along with several parameters
    # - the size of images to generate, the number of timesteps in the diffusion process,
    # and a choice between the L1 and L2 norms.

    diffusion = GaussianDiffusion(
        model,
        image_size = 128,
        timesteps = 50,   # number of steps
        loss_type = 'l1'    # L1 or L2
    )

    # We generate random data to train on, and then train the Diffusion Model in the usual fashion:

    # training_images = torch.randn(8, 3, 128, 128)
    # loss = diffusion(training_images)
    # loss.backward()

    # Once the model is trained, we can finally generate images by using the sample()
    # method of the diffusion object. Here we generate 4 images, which are only noise
    # given that our training data was random:

    # sampled_images = diffusion.sample(batch_size = 4)
    # torch.save(diffusion, "random_diffusion_model.zip")
    # diffusion.load_state_dict(torch.load("random_diffusion_model.zip"))

    from torchvision.utils import save_image
    # save_image(sampled_images, 'sampled_images_random_diffusion_model.png')

    #################################################################################################

    from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
    # CIFAR
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar.load_data()
    # x_train = np.asarray(x_train)
    # x_train = x_train.astype(np.float32)/255           # images are normalized from 0 to 1
    # new_train = torch.from_numpy(np.swapaxes(x_train,1,3))
    # n_samples, n_channels, image_size, _ = new_train.shape
    # for i in range(new_train.shape[0]):
    #     save_image(new_train[i]/255, "cifar/{}.png".format(i))


    # MNIST
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.asarray(x_train)
    x_train = x_train.astype(np.float32)/255           # images are normalized from 0 to 1
    training_images = torch.from_numpy(x_train)
    n_samples, image_size, _ = training_images.shape
    # for i in range(n_samples):
    #     save_image(training_images[i], "mnist/{}.png".format(i))


    model = Unet(
        dim = 16,
        dim_mults = (1, 2, 4),
        learned_sinusoidal_cond=False,
    )#.cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size = image_size,
        timesteps = 100,     # number of steps
        loss_type = 'l1'    # L1 or L2
    )#.cuda()


    trainer = Trainer(
        diffusion,
        "mnist/",
        # training_images,
        train_batch_size = 128,
        train_lr = 1e-4,
        train_num_steps = 100,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.9999,                # exponential moving average decay
        amp = True                        # turn on mixed precision
    )

    # trainer.train()
    # torch.save(diffusion.state_dict(), "mnist_diffusion_model.zip")
    diffusion.load_state_dict(torch.load("mnist_diffusion_model.zip"))
    sampled_images = diffusion.sample(batch_size = 4)
    save_image(sampled_images, '../sampled_images_mnist_diffusion_model.png')













