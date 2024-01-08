import torch
import torch.nn as nn
import torch.nn.functional as F

def train_epoch(net, real_x, optimizer, opt):
    batch_size = real_x.shape[0]
    real_y = torch.ones(batch_size).to(real_x.device)
    fake_y = torch.zeros(batch_size).to(real_x.device)

    fake_x1, kld = net(real_x)

    # sample from the prior distribution
    sample_z = torch.randn(batch_size, opt.latent_dim).to(real_x.device)
    fake_x2 = net.generator(sample_z)

    # compute D(x) for real and fake images along with their features
    realness_r, features_r = net.discriminator(real_x)
    realness_f1, features_f1 = net.discriminator(fake_x1)
    realness_f2, features_f2 = net.discriminator(fake_x2)

    #-------------- train the discriminator --------------#
    loss_D = F.binary_cross_entropy(realness_r, real_y) + \
        0.5 * (F.binary_cross_entropy(realness_f1, fake_y) + \
               F.binary_cross_entropy(realness_f2, fake_y))

    optimizer.zero_grad()
    loss_D.backward(retain_graph=True)
    optimizer.step()

    #-------------- train the encoder and generator --------------#
    loss_GD = F.binary_cross_entropy(realness_f2, real_y)
    loss_G = 0.5 * (0.01 * (fake_x1 - real_x).pow(2).sum() + (features_f1 - features_r.detach()).pow(2).sum()) / batch_size

    optimizer.zero_grad()
    (opt.w_kld * kld + opt.w_loss_g * loss_G + opt.w_loss_gd * loss_GD).backward()
    optimizer.step()

    return loss_D.item(), loss_G.item(), loss_GD.item(), kld.item()


