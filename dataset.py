import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_data_loader(opt):
    my_transform = transforms.Compose([
        transforms.Resize((opt.img_size, opt.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.ImageFolder(root = 'data/images', transform=my_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    return train_loader