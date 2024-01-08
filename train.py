import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torchvision.utils as utils
import numpy as np
import random
import os
import argparse
from dataset import get_data_loader
from model import VAE_GAN
from help_function import train_epoch

#--------------------- create the dirs ------------------------#
save_path = "./saved_checkpoints"
if not os.path.exists(save_path):
    os.mkdir(save_path)

#--------------------- set the parameters ---------------------#
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image size")
parser.add_argument("--num_workers", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--w_kld", type=float, default=1, help="weight for KLD loss")
parser.add_argument("--w_loss_g", type=float, default=0.01, help="weight for generator loss")
parser.add_argument("--w_loss_gd", type=float, default=1, help="weight for generator discriminator loss")
parser.add_argument("--resume_training", type=bool, default=False, help="resume training from saved model")
parser.add_argument("--is_train", type=bool, default=True, help="train or test")

opt = parser.parse_args()
print(opt)

#--------------------- set the random seeds ---------------------#
manual_seed = 114514
random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)


#--------------------- set the device ---------------------#
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#--------------------- load and process data ---------------------#
train_loader = get_data_loader(opt)

#--------------------- define the model and optimizer ---------------------#
net = VAE_GAN(opt).to(device)
optimizer = Adam(net.parameters(), lr=opt.lr)

if opt.resume_training:
    checkpoint = torch.load(os.path.join(save_path, 'checkpoint.pth'))
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
else:
    start_epoch = 0

#--------------------- train the model ---------------------#
def training():
    for epoch in range(start_epoch, opt.epochs):
        net.train()
        loss_D = []
        loss_G = []
        loss_GD = []
        loss_kld = []

        for real_x, _ in train_loader:
            real_x = real_x.to(device)
            loss_D, loss_G, loss_GD, loss_kld = train_epoch(net, real_x, optimizer, opt)
            loss_D.append(loss_D), loss_G.append(loss_G), loss_GD.append(loss_GD), loss_kld.append(loss_kld)
            loss_D = np.mean(loss_D)
            loss_G = np.mean(loss_G)
            loss_GD = np.mean(loss_GD)
            loss_kld = np.mean(loss_kld)

            generate_samples(f"./results/{epoch}.png")
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{save_path}/checkpoint.pth")


def generate_samples(save_path):
    sample_z = torch.randn(opt.num_samples, opt.latent_dim).to(device)
    with torch.no_grad():
        net.eval()
        x_hat = net.generator(sample_z)
    utils.save_image(x_hat, save_path, nrow=6, normalize=True)

if __name__ == '__main__':
    if opt.is_train:
        training()
    else:
        checkpoint = torch.load(os.path.join(save_path, 'checkpoint.pth'))
        net.load_state_dict(checkpoint['model_state_dict'])
        generate_samples("./results/test.png")