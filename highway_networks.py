#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sebastien + code de l'exemple MNIST PyTorch + implém. HN par c0nn3r
"""

import argparse
import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

class HighwayMLP(nn.Module):
    def __init__(self,
                 input_size,
                 gate_bias=-2,
                 activation_function=nn.functional.tanh,
                 gate_activation=nn.functional.softmax):

        super(HighwayMLP, self).__init__()

        self.activation_function = activation_function
        self.gate_activation = gate_activation

        self.normal_layer = nn.Linear(input_size, input_size)

        self.gate_layer = nn.Linear(input_size, input_size)
        self.gate_layer.bias.data.fill_(gate_bias)

    def forward(self, x):

        normal_layer_result = self.activation_function(self.normal_layer(x))
        gate_layer_result = self.gate_activation(self.gate_layer(x))

        multiplyed_gate_and_normal = torch.mul(normal_layer_result, gate_layer_result)
        multiplyed_gate_and_input = torch.mul((1 - gate_layer_result), x)

        return torch.add(multiplyed_gate_and_normal,
                         multiplyed_gate_and_input)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.3, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--highway-number', type=int, default=10, metavar='N',
                    help='how many highway layers to use in the model')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

    
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Lambda(lambda x: x.numpy().flatten()),
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.numpy().flatten()),
    ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

class HN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HN, self).__init__()
        self.highway_nworks = nn.ModuleList([HighwayMLP(input_dim, \
            activation_function=F.tanh) for _ in range(args.highway_number)])
        self.linear = nn.Linear(input_dim, output_dim)
 
    def forward(self, x):
        for nwork in self.highway_nworks:
            x = nwork(x)
        x = F.softmax(self.linear(x))
        return x
  
if __name__ == '__main__':
    model = HN(784, 10)
    if args.cuda:
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, \
                                                      momentum=args.momentum)
    def train(n_epoch, model, optimizer):
        model.train()
        for batch_index, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            ''' volatile -> l'historique de data n'est pas sauvegardé '''
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            yhat = model(data)
            loss = F.nll_loss(yhat, target)
            loss.backward()
            optimizer.step()
            if batch_index % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.\
                      format(n_epoch, batch_index * len(data), len(\
                      train_loader.dataset), 100. * batch_index / len(\
                                                  train_loader), loss.data[0]))
    def test(model):
        model.eval()
        loss_test = 0
        acc = 0
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            yhat = model(data)
            loss_test += F.nll_loss(yhat, target).data[0]
            pred = yhat.data.max(1)[1]
            acc += pred.eq(target.data).cpu().sum()
        loss_test = loss_test
        loss_test /= len(test_loader)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.\
              format(loss_test, acc, len(test_loader.dataset),\
              100. * acc / len(test_loader.dataset)))
    
    
    for n_epoch in range(1, args.epochs + 1):
        tmps1 = time.time()
        train(n_epoch, model, optimizer)
        tmps2=time.time()-tmps1
        print ('Temps d\'exécution = {}s'.format(tmps2))
        tmps1 = time.time()
        test(model)
        tmps2=time.time()-tmps1
        print ('Temps d\'exécution = {}s'.format(tmps2))