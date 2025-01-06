import matplotlib.pyplot as plt
import torch
from torch.onnx.symbolic_opset8 import zeros_like
import copy
from CKA import CudaCKA



from auto_load_data import *
from auto_model import *
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics.functional import multiclass_f1_score, multiclass_accuracy, multilabel_accuracy
from datetime import datetime
import numpy as np
from torchinfo import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda_cka = CudaCKA(device)

model = NNClassifier().to(device)
training_loader, validation_loader = get_data_loaders()
loss_fn = torch.nn.CrossEntropyLoss()

# Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)



def find_best_first_layer(model):
    print('find best first layer')
    print(len(model.conv1_layers))
    input = model.input[-1]
    scores = []
    cka_scores = []
    for i in range(len(model.conv1_layers)):
        output = model.conv1_layers[i](input)
        print('layer {}'.format(i))
        print(input.shape, output.shape)
        v = nn.Parameter(torch.where(model.conv1_layers[i].weight.abs() > 0, 1., 0.))
        with torch.no_grad():
            input_reshape_layer = nn.Conv2d(3, 6, 4, device=device)
            input_reshape_layer.weight = v
        input_reshaped = input_reshape_layer(input)
        # print(v[:5, :5])
        # print(input_reshape_layer.weight)
        print(output[0, 0, :3, :3])
        print(input_reshaped[0, 0, :3, :3])
        print(input_reshaped.shape, output.T.shape)
        score = cuda_cka.kernel_CKA(output[0, 0], input_reshaped[0, 0])
        scores.append(score)

        # v = torch.where(output > 0, 1, 0)
        # reshaped_input = torch.autograd.grad(output, inputs=(input.requires_grad_(True),), grad_outputs=v, allow_unused=True)[0]
        # score = torch.cdist(output, input)
        # output_model = model.clone()
        # output_model.first_layer_used = i
        # cka_scores.append(CKA(model1=input_model, model2=output_model))
        # print(reshaped_input)

        # scores.append((input.shape, output.shape, type(reshaped_input)))
    print('cka scores: ' + str(scores))
    return scores

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    running_accuracy = 0.
    last_accuracy = 0.
    running_f1_score = 0.
    last_f1_score = 0.
    accuracy = 0.
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data[0].to(device), data[1].to(device)
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

        # Compute Accuracy
        last_accuracy = multiclass_accuracy(outputs, labels)*4
        accuracy += last_accuracy

        if i % 1000 == 999:
            last_f1_score = running_f1_score / 1000

            accuracy = accuracy / 4000
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            print('  batch {} accuracy: {}'.format(i + 1, accuracy))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss, last_f1_score, accuracy.cpu().numpy().round(2)

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/CIFAR_10_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 3

best_vloss = 1_000_000.
avg_losses = []
avg_f1_scores = []
avg_accuracies = []
avg_vaccuracies = []
avg_vlosses = []

print('Auto-training')
for epoch in range(EPOCHS):
    # print(summary(model, input_size=(4, 3, 32, 32)))
    print('Num of first layers: {}'.format(len(model.conv1_layers)))
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.add_layer()

    model.train(mode=True)

    print('model input length: ' + str(len(model.input)))
    if epoch_number > 0:
        print('finding')
        layer_vals = find_best_first_layer(model)
        print('layer_vals: {}'.format(layer_vals))
    avg_loss, avg_f1_score, accuracy = train_one_epoch(epoch_number, writer)
    avg_losses.append(avg_loss)
    avg_f1_scores.append(avg_f1_score)
    avg_accuracies.append(accuracy)


    running_vloss = 0.0
    running_vaccuracy = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata[0].to(device), vdata[1].to(device)
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            vaccuracy = multiclass_accuracy(voutputs, vlabels)
            running_vloss += vloss
            running_vaccuracy += vaccuracy

    avg_vloss = running_vloss / (i + 1)
    avg_vlosses.append(avg_vloss.cpu().numpy())
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
    avg_vaccuracy = running_vaccuracy / (i + 1)
    avg_vaccuracies.append(avg_vaccuracy.cpu().numpy())
    print('ACCURACY train {} valid {}'.format(avg_accuracies[-1], avg_vaccuracy))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss},
                    epoch_number + 1)

    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        # model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        # torch.save(model.state_dict(), model_path)

    epoch_number += 1





fig, ax = plt.subplots(1, 2)
fig.suptitle('Auto Training vs. Validation')

ax[0].plot(avg_losses, label='Training')
ax[0].plot(avg_vlosses, label='Validation')
ax[0].set_title('Loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].legend()

ax[1].plot(avg_accuracies, label='Training')
ax[1].plot(avg_vaccuracies, label='Validation')
ax[1].set_title('Accuracy')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')
ax[1].legend()

plt.show()