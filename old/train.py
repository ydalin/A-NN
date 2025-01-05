from load_data import *
from model import *
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics.functional import multiclass_f1_score, multiclass_accuracy
from datetime import datetime
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = NNClassifier().to(device)
training_loader, validation_loader = get_data_loaders()
loss_fn = torch.nn.CrossEntropyLoss()

# Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    running_accuracy = 0.
    last_accuracy = 0.
    running_f1_score = 0.
    last_f1_score = 0.

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
        running_f1_score += multiclass_f1_score(outputs, labels)
        running_accuracy += multiclass_accuracy(outputs, labels)

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_accuracy = running_accuracy / 1000
            last_f1_score = running_f1_score / 1000
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss, last_f1_score, last_accuracy

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/CIFAR_10_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.
avg_losses = []
avg_f1_scores = []
avg_accuracies = []
avg_vlosses = []

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss, avg_f1_score, avg_accuracy = train_one_epoch(epoch_number, writer)
    avg_losses.append(avg_loss)
    avg_f1_scores.append(avg_f1_score.cpu().numpy())
    avg_accuracies.append(avg_accuracy.cpu().numpy())


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata[0].to(device), vdata[1].to(device)
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    avg_vlosses.append(avg_vloss.cpu().numpy())
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1




plt.plot(avg_losses, label='Training Loss')
# plt.plot(avg_f1_scores, label='Average F1 Score')
plt.plot(avg_vlosses, label='Validation Loss')
plt.title('Training vs. Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Scores')
plt.legend()
plt.show()