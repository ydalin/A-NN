import torchvision.transforms as transforms, torchvision, matplotlib.pyplot as plt
import torch
from torch.utils.data import random_split


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_data_loaders(show_imgs=False):
    all_transforms = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    transforms.Resize(size=(32, 32)),
                                    transforms.Grayscale(1),
                                    ])

    dataset_CIFAR10 = torchvision.datasets.CIFAR10(root='./data',
                                            train=True,
                                            download=True,
                                            transform=all_transforms)

    dataset = dataset_CIFAR10

    train_data, val_data = random_split(dataset_CIFAR10, [0.8, 0.2])

    training_loader = torch.utils.data.DataLoader(train_data,
                                              batch_size=4,
                                              shuffle=True)

    validation_loader = torch.utils.data.DataLoader(val_data,
                                              batch_size=4,
                                              shuffle=True)

    if show_imgs:
        images, labels = next(iter(training_loader))
        plt.imshow(torchvision.utils.make_grid(images).permute(1, 2, 0) / 2 + 0.5)
        plt.title(' '.join(dataset.classes[label] for label in labels)); plt.show()
    return training_loader, validation_loader