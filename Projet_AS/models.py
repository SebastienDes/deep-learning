#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 18:51:00 2018

@author: seb
"""

#import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    """
    Cette classe contient la structure du réseau de neurones
    """

    def __init__(self):
        super(ConvNet, self).__init__()
        # On défini d'abord les couches de convolution et de pooling comme un
        # groupe de couches `self.features`
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, (5, 5), stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2, padding=0),
            nn.Conv2d(32, 64, (5, 5), stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2, padding=0),
            nn.Conv2d(64, 64, (5, 5), stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2, padding=0, ceil_mode=True),
        )
        # On défini les couches fully connected comme un groupe de couches
        # `self.classifier`
        self.classifier = nn.Sequential(
            nn.Linear(64*4*4, 1000),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1000, 4)
            # Rappel : Le softmax est inclus dans la loss, ne pas le mettre ici
        )

    # méthode appelée quand on applique le réseau à un batch d'input
    def forward(self, input):
        bsize = input.size(0) # taille du batch
        output = self.features(input) # on calcule la sortie des conv
        output = output.view(bsize, -1) # on applati les feature map 2D en un
                                        # vecteur 1D pour chaque input
        output = self.classifier(output) # on calcule la sortie des fc
        return output
    
class Alexnet(nn.Module):

    def __init__(self, alex):

        super(Alexnet, self).__init__()
        self.features = nn.Sequential( *list(alex.features.children()))
        self.classifier = nn.Sequential( *list(alex.classifier.children())[:-1])
    def forward(self, x):
        print(x.size())
        x = self.features(x)
        print(x.size())
        x = x.view(x.size(0), -1)
        print(x.size())
        x = self.classifier(x)
        print(x.size())
        return x

class Conv3ThreeBlocks(nn.Module):
    def __init__(self, conv3):
        super(Conv3ThreeBlocks, self).__init__()
        self.block1 = nn.Sequential(*list(conv3.block1.children()))
        self.block2 = nn.Sequential(*list(conv3.block2.children()))
        self.block3 = nn.Sequential(*list(conv3.block3.children()))
        self.classifier = nn.Sequential(
                nn.Linear(384, 200),
                nn.BatchNorm1d(200),
                nn.ReLU(),
                nn.Linear(200, 200),
                nn.BatchNorm1d(200),
                nn.ReLU(),
                nn.Linear(200, 10)
        )
        
    def forward(self, input):
        bsize = input.size(0)
        output = self.block1(input)
        output = F.max_pool2d(output, 2)
        output = self.block2(output)
        output = F.max_pool2d(output, 2)
        output = self.block3(output)
        output = F.avg_pool2d(output, 2)
        output = output.view(bsize, -1)
        output = self.classifier(output)
        return output
    
class Conv3TwoBlocks(nn.Module):
    def __init__(self, conv3):
        super(Conv3TwoBlocks, self).__init__()
        self.block1 = nn.Sequential(*list(conv3.block1.children()))
        self.block2 = nn.Sequential(*list(conv3.block2.children()))
        self.classifier = nn.Sequential(
                nn.Linear(256, 200),
                nn.BatchNorm1d(200),
                nn.ReLU(),
                nn.Linear(200, 200),
                nn.BatchNorm1d(200),
                nn.ReLU(),
                nn.Linear(200, 10)
        )
        
    def forward(self, input):
        bsize = input.size(0)
        output = self.block1(input)
        output = F.max_pool2d(output, 2)
        output = self.block2(output)
        output = F.avg_pool2d(output, 4)
        output = output.view(bsize, -1)
        output = self.classifier(output)
        return output

class Conv3(nn.Module):
    def __init__(self):
        super(Conv3, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 96, (11,11), stride=4, padding=5),
            nn.ReLU(),
            nn.Conv2d(96, 96, (1, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(96, 96, (1, 1), stride=1, padding=0),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(96, 256, (5, 5), stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(256, 256, (1, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, (1, 1), stride=1, padding=0),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 384, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, (1, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(384, 384, (1, 1), stride=1, padding=0),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
                nn.Linear(384, 4),
        )
        
    def forward(self, input):
        bsize = input.size(0)
        output = self.block1(input)
        output = F.max_pool2d(output, 2)
        output = self.block2(output)
        output = F.max_pool2d(output, 2)
        output = self.block3(output)
        output = F.max_pool2d(output, 2)
        output = output.view(bsize, -1)
        output = self.classifier(output)
        return output

class Conv4(nn.Module):
    """
    Cette classe contient la structure du réseau de neurones
    """

    def __init__(self):
        super(Conv4, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 96, (11,11), stride=4, padding=5),
            nn.ReLU(),
            nn.Conv2d(96, 96, (1, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(96, 96, (1, 1), stride=1, padding=0),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(96, 256, (5, 5), stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(256, 256, (1, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, (1, 1), stride=1, padding=0),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 384, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, (1, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(384, 384, (1, 1), stride=1, padding=0),
            nn.ReLU()
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(384, 1024, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, (1,1), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, (1,1), stride=1, padding=0),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 4)
        )

    def forward(self, input):
        bsize = input.size(0)
        output = self.block1(input)
        output = F.max_pool2d(output, 2)
        output = self.block2(output)
        output = F.max_pool2d(output, 2)
        output = self.block3(output)
        output = F.max_pool2d(output, 2)
        output = self.block4(output)
        output = F.avg_pool2d(output, 1)
        output = output.view(bsize, -1)
        output = self.classifier(output)
        return output