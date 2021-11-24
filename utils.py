import torch
import torchvision
from torchvision import transforms
from config import BATCH_SIZE


def get_dataset():
    # Define the composer.
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Download & Load in Training/Testing dataset.
    train_set = torchvision.datasets.CIFAR10(
        root='./data/vgg16/', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(
        root='./data/vgg16', train=False, download=True, transform=transform)
    return train_set, test_set

def get_data_loader():
    train_set , test_set = get_dataset()

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, test_loader
