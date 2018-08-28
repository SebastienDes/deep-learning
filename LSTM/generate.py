#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 21:50:33 2017

@author: seb
"""

import torch
from torch.autograd import Variable
from charDataset import char_tensor
import string

def generate(decoder, prime_str='A', predict_len=100, temperature=0.8, 
                                         cuda=False):
    all_characters = string.printable
    if cuda:
        hidden = decoder.init_hidden(1, cuda=True)
        prime_input = Variable(char_tensor(prime_str).unsqueeze(0)).cuda()
    else:
        hidden = decoder.init_hidden(1)
        prime_input = Variable(char_tensor(prime_str).unsqueeze(0))

    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[:,p], hidden)
        
    inp = prime_input[:,-1]
    
    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = Variable(char_tensor(predicted_char).unsqueeze(0))
        if cuda:
            inp = inp.cuda()

    return predicted