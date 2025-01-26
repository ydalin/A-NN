import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cka import CKA # https://github.com/AntixK/PyTorch-Model-Compare
import numpy as np

from CKA import CudaCKA


# PyTorch models inherit from torch.nn.Module
class NNClassifier(nn.Module):
    def __init__(self):
        super(NNClassifier, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.input = []
        self.similarities = []
        self.conv1_layers = torch.nn.ModuleList([nn.Conv2d(1 , 6, 4)])
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
        self.conv1_layers.append(nn.Conv2d(1, 6, 4, device='cuda:0').requires_grad_(False))
        self.deactivate_layer(-1) # deactivate the newly added layer
        return 0

    def random_first_layer(self):
        self.first_layer_used = torch.randint(len(self.conv1_layers), size=(1,))

    # def get_similarities(self, x):
    #     outputs = []
    #     for layer in self.conv1_layers:
    #         outputs.append(layer(x))
    #     self.similarities
    #

    def deactivate_layer(self, num_layer):
        layer = self.conv1_layers[num_layer]
        for p in layer.parameters():
            p.requires_grad_(False)

    def activate_layer(self, num_layer):
        layer = self.conv1_layers[num_layer]
        for p in layer.parameters():
            p.requires_grad_(True)

    def earth_mover_distance(self, num_layer, layer_input):
        """
        Distance between input and output of layer in first set of layers
        :param num_layer: int: location in list of first set of layers
        :return: float: Earth mover distance
        """
        with torch.no_grad():
            v = nn.Parameter(torch.where(self.conv1_layers[num_layer].weight.abs() > 0, 1., 0.))
            input_reshape_layer = nn.Conv2d(1, 6, 4, device=self.device)
            input_reshape_layer.weight = v
            input_reshaped = input_reshape_layer(layer_input)
            model_layer = self.conv1_layers[num_layer]
            model_layer_output = model_layer(layer_input)
        return torch.mean(torch.square(torch.cumsum(input_reshaped, dim=-1) - torch.cumsum(model_layer_output, dim=-1)),
                          dim=(0, 1, 2, 3))

    def CKA_layer_weights_method(self, layer_num, layer_input):
        np_cka = CudaCKA(device=self.device)
        layer = self.conv1_layers[layer_num]
        output = layer(layer_input)
        layer_weight = layer.weight
        print('before, after')
        print(layer_weight.shape, output.shape)
        layer_weight = layer_weight.permute(*torch.arange(layer_weight.ndim - 1, 0, -1))
        print(layer_weight.shape, output.shape)

        result = np_cka.kernel_CKA(layer.weight, output)
        return result




class NNInput(nn.Module):
    def __init__(self):
        super(NNInput, self).__init__()
        self.conv1_layer = nn.Conv2d(1, 6, 4)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1_layer(x)))
        return x







