# -*- coding: utf-8 -*-

import numpy as np
import time
import argparse
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from models import Conv3
from models import Conv4
from models import Alexnet
from models import ConvNet

from utils import *
from torchsample.transforms import Rotate

PRINT_INTERVAL = 50
CUDA = False

def get_dataset(path):
    """ Permet de récupérer le jeu de données """
    transform = transforms.Compose(
        [transforms.ToTensor(), 
         transforms.Normalize((0.491, 0.482, 0.447), (0.202, 0.199, 0.201))
         ])

    transform2 = transforms.Compose([transforms.ToTensor(), 
        transforms.Normalize((0.491, 0.482, 0.447), (0.202, 0.199, 0.201))
        ])

    trainset = torchvision.datasets.CIFAR10(path, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, pin_memory=CUDA, num_workers=2)

    testset = torchvision.datasets.CIFAR10(path, train=False,
                                           download=True, transform=transform2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, pin_memory=CUDA, num_workers=2)
    return trainloader, testloader

def get_accuracy(output, target):
    """ Calcule l'accuracy (précision) moyenne sur le batch """
    batch_size = target.size(0)

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)
    res = correct_k.mul_(100.0 / batch_size)
    return res

def get_loss(outputs):
    """ Calcule la loss pour notre prédiction de rotation. """
    total_loss = 0.0
    for y in range(len(outputs)):
        total_loss -= torch.mean(outputs[y])
    total_loss /= len(outputs)
    return total_loss
        
def epoch(data, model, criterion, optimizer=None):
    """
    Fait une passe (appelée epoch en anglais) sur les données `data` avec le
    modèle `model`. Evalue `criterion` comme loss.
    Si `optimizer` est fourni, effectue une epoch d'apprentissage en utilisant
    l'optimiseur donné, sinon, effectue une epoch d'évaluation (pas de backward)
    du modèle.
    """
    # indique si le modele est en mode eval ou train (certaines couches se
    # comportent différemment en train et en eval)
    model.eval() if optimizer is None else model.train()

    global loss_plot

    # on itere sur les batchs du dataset
    for i, (input, _) in enumerate(data):
        
        # forward
        target = torch.LongTensor([0]*len(input))
        target90 = torch.LongTensor([1]*len(input))
        target180 = torch.LongTensor([2]*len(input))
        target270 = torch.LongTensor([3]*len(input))
        input90 = torch.FloatTensor(input.size()).zero_()
        input180 = torch.FloatTensor(input.size()).zero_()
        input270 = torch.FloatTensor(input.size()).zero_()
        for k in range(len(input)):
            input90[k] = Rotate(90)(input[k])
            input180[k] = Rotate(180)(input[k]) 
            input270[k] = Rotate(270)(input[k]) 
        if CUDA: # si on fait du GPU, passage en CUDA
            input = input.cuda()
            input90 = input90.cuda()
            input180 = input180.cuda()
            input270 = input270.cuda()
            target = target.cuda()
            target90 = target90.cuda()
            target180 = target180.cuda()
            target270 = target270.cuda()
        output = model(Variable(input))
        output90 = model(Variable(input90))
        output180 = model(Variable(input180))
        output270 = model(Variable(input270))
#        loss = get_loss([output, output90, output180, output270])
        loss = criterion(output, Variable(target))
        loss += criterion(output90, Variable(target90))
        loss += criterion(output180, Variable(target180))
        loss += criterion(output270, Variable(target270))
        loss /= 4

        # backward si on est en "train"
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # calcul des metriques
        prec0 = get_accuracy(output.data, target)
        prec90 = get_accuracy(output90.data, target90)
        prec180 = get_accuracy(output180.data, target180)
        prec270 = get_accuracy(output270.data, target270)
        prec = sum([prec0, prec90, prec180, prec270])/4

        # affichage des infos
        if i % PRINT_INTERVAL == 0:
#            print('Loss : {}'.format(loss.data))
#            print('Accuracy 0 : {}'.format(prec0))
#            print('Accuracy 90 : {}'.format(prec90))
#            print('Output90 :{}'.format(output90))
#            print('Accuracy 180 : {}'.format(prec180))
#            print('Accuracy 270 : {}'.format(prec270))
            if optimizer is None:
                print('Accuracy test: {}'.format(prec))
                
    return


def main(params):
    # define model, loss, optim
    model = Conv3()
    #model = Conv4()
    #model = ConvNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)#, 
                                              #weight_decay=0.0005)
    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, 
                                                           gamma=0.2)

    if CUDA: # si on fait du GPU, passage en CUDA
        model = model.cuda()
        criterion = criterion.cuda()

    # On récupère les données
    train, test = get_dataset(params.path)

    # On itère sur les epochs
    for i in range(params.epochs):
        print("=================\n=== EPOCH "+str(i+1)+" =====\n=================\n")
        # Phase de train
        lr_sched.step()
        epoch(train, model, criterion, optimizer) 
        # Phase d'evaluation
        epoch(test, model, criterion)
        # sauvegarde du modèle à la dernière epoch
        if i==params.epochs-1:
            torch.save(model, './trained_model{}.pth'.format(i))

if __name__ == '__main__':
    
    # Paramètres en ligne de commande
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='/tmp/datasets/cifar10', type=str, 
                                        metavar='DIR', help='path to dataset')
    parser.add_argument('--epochs', default=5, type=int, metavar='N', 
                                        help='number of total epochs to run')
    parser.add_argument('--cuda', dest='cuda', action='store_true', 
                                            help='activate GPU acceleration')

    args = parser.parse_args()
    if args.cuda:
        CUDA = True
        cudnn.benchmark = True
        
    main(args)

    input("done")