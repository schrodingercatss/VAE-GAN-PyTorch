import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Encoder, Generator, Discriminator


class VAE_GAN(nn.Module):
    def __init__(self, opt):
        self.encoder = Encoder(opt.latent_dim)
        self.generator = Generator(opt.latent_dim)
        self.discriminator = Discriminator()

    def forward(self, x):
        z, kld = self.encoder(x)
        x_hat = self.generator(z)
        return x_hat, kld
