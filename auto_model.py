import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cka import CKA # https://github.com/AntixK/PyTorch-Model-Compare
import numpy as np



# PyTorch models inherit from torch.nn.Module
class NNClassifier(nn.Module):
    def __init__(self):
        super(NNClassifier, self).__init__()
        self.input = []
        self.similarities = []
        self.conv1_layers = torch.nn.ModuleList([nn.Conv2d(3, 6, 4)])
        self.first_layer_used = 0
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 4)
        self.fc1 = nn.Linear(400, 200)
        self.fc2 = nn.Linear(200, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        self.get_input(x)
        x = self.pool(F.relu(self.conv1_layers[self.first_layer_used](x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_input(self, x):
        if len(self.input) == 0:
            self.input.append(x)
        else:
            self.input[0] = x

    def get_input_layer(self, x):
        if x.shape[1] == 1:
            x = torch.stack([x, x, x], dim=2)

    def add_layer(self):
        self.conv1_layers.append(nn.Conv2d(3, 6, 4, device='cuda:0'))
        return 0

    def random_first_layer(self):
        self.first_layer_used = torch.randint(len(self.conv1_layers), size=(1,))

    # def get_similarities(self, x):
    #     outputs = []
    #     for layer in self.conv1_layers:
    #         outputs.append(layer(x))
    #     self.similarities
    #


class NNInput(nn.Module):
    def __init__(self):
        super(NNInput, self).__init__()
        self.conv1_layer = nn.Conv2d(3, 6, 4)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1_layer(x)))
        return x







