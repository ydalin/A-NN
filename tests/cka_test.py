import torch_cka as cka
import torch
import torchvision
import torchvision.transforms as transforms, torchvision, matplotlib.pyplot as plt
from setuptools.sandbox import save_path
from torch.utils.data import random_split, SequentialSampler, DataLoader

# from auto_load_data import device
device = "cuda:0" if torch.cuda.is_available() else "cpu"


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

train_data, val_data = random_split(dataset_CIFAR10, [.9, .1])


training_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=4,
                                              shuffle=True)

subset_indices = [0, 1, 2, 3] # select your indices here as a list
subset = torch.utils.data.Subset(dataset, subset_indices)
cka_testing_loader = torch.utils.data.DataLoader(subset, batch_size=4, num_workers=0, shuffle=False)



input = torch.randn(1, 1, 224, 224)
nn1 = torch.nn.Conv2d(1, 10, 3)
nn2 = torch.nn.Conv2d(1, 10, 3)

out_1 = nn1(input)
out_2 = nn2(input)

# the small sample size in cka testing loader results in a false high CKA score even for dissimilar data
#the full train dataset works


comparison = cka.CKA(nn2, nn1, "nn2", "nn1", device=device)
comparison.compare(training_loader, training_loader)
comparison.plot_results('cka_test.png', 'cka test')
results = comparison.export()

comparison2 = cka.CKA(nn1, nn1, "nn1", "nn1", device=device)
comparison2.compare(training_loader, training_loader)
comparison2.plot_results('cka_test.png', 'cka test')
results2 = comparison2.export()
print('results')
print(results)
print('results2')
print(results2)
