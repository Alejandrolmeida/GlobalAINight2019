import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.onnx as onnx
import torch.optim as optim
import torch.nn.functional as F

from time import time
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from azureml.core.run import Run

###################################################################
# Helpers                                                         #
###################################################################
def info(msg, char = "#", width = 75):
    print("")
    print(char * width)
    print(char + "   %0*s" % ((-1*width)+5, msg) + char)
    print(char * width)

def check_dir(path, check=False):
    if check:
        assert os.path.exists(path), '{} does not exist!'.format(path)
    else:
        if not os.path.exists(path):
            os.makedirs(path)
        return Path(path).resolve()

###################################################################
# Data Loader                                                     #
###################################################################
def get_dataloader(train=True, batch_size=64, data_dir='data'):
    transform = transforms.Compose([transforms.RandomRotation(10), 
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                              ])
    dataset = datasets.MNIST('train', download=True, train=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    return loader

###################################################################
# Saving                                                          #
###################################################################
def save_model(model, device, path, name, no_cuda):
    if not no_cuda:
        x = torch.randint(255, (1, 28*28), dtype=torch.float).to(device) / 255
    else:
        x = torch.randint(255, (1, 28*28), dtype=torch.float) / 255

    onnx.export(model, x, "./outputs/{}.onnx".format(name))
    print('Saved onnx model to model.onnx')

    torch.save(model.state_dict(), "./outputs/{}.pth".format(name))
    print('Saved PyTorch Model to model.pth')

###################################################################
# Models                                                          #
###################################################################
class SimpleLinear(nn.Module):
    def __init__(self):
        super(SimpleLinear, self).__init__()
        self.layer1 = nn.Linear(28*28, 10)

    def forward(self, x):
        x = self.layer1(x)
        return F.softmax(x, dim=1)

class NeuralNework(nn.Module):
    def __init__(self):
        super(NeuralNework, self).__init__()
        self.layer1 = nn.Linear(28*28, 512)
        self.layer2 = nn.Linear(512, 512)
        self.output = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.output(x)
        return F.softmax(x, dim=1)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

###################################################################
# Train/Test                                                      #
###################################################################
def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    train_log = defaultdict(list)
    t_log = time()
    n_samples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        t0 = time()
        data, target = data.to(device), target.to(device).long()
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        t1 = time()
        loss.backward()
        t2 = time()
        optimizer.step()
        t3 = time()
        n_samples += data.shape[0]
        if batch_idx % log_interval == 0:
            pred = output.max(1, keepdim=True)[1]
            correct = pred.eq(target.view_as(pred)).sum().item()

            train_log['n_iter'].append(epoch * len(train_loader) + batch_idx + 1)
            train_log['n_samples'].append(n_samples + (epoch - 1) * len(train_loader.dataset))
            train_log['loss'].append(loss.detach())
            train_log['accuracy'].append(100. * correct / data.shape[0])
            train_log['time_batch'].append(t3 - t0)
            train_log['time_batch_forward'].append(t1 - t0)
            train_log['time_batch_backward'].append(t2 - t1)
            train_log['time_batch_update'].append(t3 - t2)
            t4 = time()
            train_log['time_batch_avg'].append((t4 - t_log) / log_interval)
            print(
                'Train Epochs: {} [{:5d}/{:5d} ({:3.0f}%)]'
                '\tLoss: {:.6f}'
                '\tTime: {:.4f}ms/batch'.format(
                    epoch, n_samples, len(train_loader.dataset),
                    100. * (batch_idx + 1) / len(train_loader), loss.item(),
                    1000 * (t4 - t_log) / log_interval,
                )
            )
            t_log = time()
    return train_log

def test(model, device, test_loader, log_interval):
    model.eval()
    test_loss = 0
    correct = 0
    preds = []
    targets = []
    num_batches = 0
    with torch.no_grad():
        for data, target in test_loader:
            num_batches += 1
            data, target = data.to(device), target.to(device).long()
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            preds.append(pred.cpu().numpy())
            targets.append(target.cpu().numpy())
    
    preds = np.concatenate(preds).squeeze()
    targets = np.concatenate(targets).squeeze()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'
        ''.format(
            test_loss,
            correct, len(test_loader.dataset), accuracy,
            )
        )
    return test_loss, accuracy

###################################################################
# Main Loop                                                       #
###################################################################
def main(data_dir, output_dir, log_dir, epochs, batch, lr, model_kind, log_interval):
    # use GPU?
    no_cuda=False
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using {} device'.format(device))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # AML Logging (if available)
    try:
        run = Run.get_context()
        print('Using AML Logging...')
        run.log('data', data_dir)
        run.log('output', output_dir)
        run.log('logs', log_dir)
        run.log('epochs', epochs)
        run.log('batch', batch)
        run.log('learning_rate', lr)
        run.log('model_kind', model_kind)
        run.log('device', device)
    except:
        run = None

    # get data loaders
    training = get_dataloader(train=True, batch_size=batch, data_dir=data_dir)
    testing = get_dataloader(train=False, batch_size=batch, data_dir=data_dir)

    # model
    if model_kind == 'linear':
        model = SimpleLinear().to(device)
    elif model_kind == 'nn':
        model = NeuralNework().to(device)
    else:
        model = CNN().to(device)

    info('Model')
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_log = defaultdict(list)
    val_log = defaultdict(list)

    train(model, device, training, optimizer, epochs, log_interval)
    test(model, device, testing, log_interval)

    info('Saving Model')
    save_model(model, device, output_dir, 'model', no_cuda)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN Training for Image Recognition.')
    parser.add_argument('-d', '--data', help='directory to training and test data', default='data')
    parser.add_argument('-o', '--output', help='output directory', default='outputs')
    parser.add_argument('-g', '--logs', help='log directory', default='logs')
    parser.add_argument('-e', '--epochs', help='number of epochs', default=5, type=int)
    parser.add_argument('-b', '--batch', help='batch size', default=64, type=int)
    parser.add_argument('-l', '--lr', help='learning rate', default=0.001, type=float)
    parser.add_argument('-m', '--model', help='model type', default='cnn', choices=['linear', 'nn', 'cnn'])
    parser.add_argument('-log', '--loginterval', help='log interval', default=10, type=int)

    args = parser.parse_args()

    args.data = check_dir(args.data).resolve()
    args.outputs = check_dir(args.output).resolve()
    args.logs = check_dir(args.logs).resolve()

    main(data_dir=args.data, output_dir=args.output, log_dir=args.logs, 
         epochs=args.epochs, batch=args.batch, lr=args.lr, model_kind=args.model, log_interval=args.loginterval)