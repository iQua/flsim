import load_data
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time
import json
import os

IMAGE_SIZE = 28

# Training settings
lr = 0.01
momentum = 0.9
log_interval = 10
rou = 1
loss_thres = 0.001

# Cuda settings
use_cuda = torch.cuda.is_available()
device = torch.device(  # pylint: disable=no-member
    'cuda' if use_cuda else 'cpu')


class Generator(load_data.Generator):
    """Generator for MNIST dataset."""

    # Extract MNIST data using torchvision datasets
    def read(self, path):
        self.trainset = {}
        self.labels = []
        trainset_size = 0

        train_dir = os.path.join(path, 'train')

        for file in os.listdir(train_dir):
            with open(os.path.join(train_dir, file)) as json_file:
                logging.info('loading {}'.format(os.path.join(train_dir, file)))
                data = json.load(json_file)
                self.trainset.update(data)
                for user in data['users']:
                    self.labels += data['user_data'][user]['y']
                    self.labels = list(set(self.labels))

                    trainset_size += len(data['user_data'][user]['y'])

        self.labels.sort()
        self.trainset_size = trainset_size

        self.testset = {}
        test_dir = os.path.join(path, 'test')

        for file in os.listdir(test_dir):
            with open(os.path.join(test_dir, file)) as json_file:
                logging.info('loading {}'.format(os.path.join(test_dir, file)))
                data = json.load(json_file)
                self.testset.update(data)


    def generate(self, path):
        self.read(path)

        return self.trainset


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def get_optimizer(model):
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum)


def get_trainloader(trainset, batch_size):
    # Convert the dictionary-format of trainset (with keys of 'x' and 'y') to
    # TensorDataset, then create the DataLoader from it
    x_train = np.array(trainset['x'], dtype=np.float32)
    x_train = np.reshape(x_train, (-1, 1, IMAGE_SIZE, IMAGE_SIZE))
    x_train = torch.Tensor(x_train)
    y_train = np.array(trainset['y'], dtype=np.int32)
    y_train = torch.Tensor(y_train).type(torch.int64)

    train_dataset = TensorDataset(x_train, y_train)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader


def get_testloader(testset, batch_size):
    # Convert the dictionary-format of testset (with keys of 'x' and 'y') to
    # TensorDataset, then create the DataLoader from it
    x_test = np.array(testset['x'], dtype=np.float32)
    x_test = np.reshape(x_test, (-1, 1, IMAGE_SIZE, IMAGE_SIZE))
    x_test = torch.Tensor(x_test)
    y_test = np.array(testset['y'], dtype=np.int32)
    y_test = torch.Tensor(y_test).type(torch.int64)

    test_dataset = TensorDataset(x_test, y_test)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return test_loader


def extract_weights(model):
    weights = []
    for name, weight in model.to(torch.device('cpu')).named_parameters():  # pylint: disable=no-member
        if weight.requires_grad:
            weights.append((name, weight.data))

    return weights


def load_weights(model, weights):
    updated_state_dict = {}
    for name, weight in weights:
        updated_state_dict[name] = weight

    model.load_state_dict(updated_state_dict, strict=False)

def flatten_weights(weights):
    # Flatten weights into vectors
    weight_vecs = []
    for _, weight in weights:
        weight_vecs.extend(weight.flatten().tolist())

    return np.array(weight_vecs)


def extract_grads(model):
    grads = []
    for name, weight in model.to(torch.device('cpu')).named_parameters():  # pylint: disable=no-member
        if weight.requires_grad:
            grads.append((name, weight.grad))

    return grads


def train(model, trainloader, optimizer, epochs, reg=None):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)

    # Get the snapshot of weights when training starts, if regularization is on
    if reg is not None:
        old_weights = flatten_weights(extract_weights(model))
        old_weights = torch.from_numpy(old_weights)

    for epoch in range(1, epochs + 1):
        for batch_id, (image, label) in enumerate(trainloader):
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)

            # Add regularization
            if reg is not None:
                new_weights = flatten_weights(extract_weights(model))
                new_weights = torch.from_numpy(new_weights)
                mse_loss = nn.MSELoss(reduction='sum')
                l2_loss = rou/2 * mse_loss(new_weights, old_weights)
                l2_loss = l2_loss.to(torch.float32)
                loss += l2_loss

            loss.backward()
            optimizer.step()
            if batch_id % log_interval == 0:
                logging.debug('Epoch: [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, epochs, loss.item()))

            # Stop training if model is already in good shape
            if loss.item() < loss_thres:
                return loss.item()

    if reg is not None:
        logging.info(
            'loss: {} l2_loss: {}'.format(loss.item(), l2_loss.item()))
    else:
        logging.info(
            'loss: {}'.format(loss.item()))
    return loss.item()


def test(model, testloader):
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    total = len(testloader.dataset)
    with torch.no_grad():
        for image, label in testloader:
            image, label = image.to(device), label.to(device)
            output = model(image)
            # sum up batch loss
            test_loss += F.nll_loss(output, label, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

    accuracy = correct / total
    logging.debug('Accuracy: {:.2f}%'.format(100 * accuracy))

    return accuracy
