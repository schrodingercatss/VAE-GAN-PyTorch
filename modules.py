import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.avgpool = nn.AvgPool2d(kernel_size=4, stride=1, padding=0)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        self.fc_mu = nn.Linear(512, out_dim)
        self.fc_logvar = nn.Linear(512, out_dim)

    def reparameterize(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
        return z, kld.mean()
    
    def forward(self, x):
        x = self.resnet(x).squeeze()
        z, kld = self.reparameterize(x)
        return z, kld
    


class Generator(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.convs = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(512, 384, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(96, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.reshape(-1, z.shape[1], 1, 1)
        x = self.convs(z)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )

        self.last_conv = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        feature_d = self.conv(x)
        realness = self.last_conv(feature_d)
        feature_d = F.avg_pool2d(feature_d, kernel_size=4, stride=1, padding=0)
        return realness.squeeze(), feature_d.squeeze()