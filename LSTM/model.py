# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTMt(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, output_dim, num_layers, 
                                                             minibatch_size):
        super(LSTMt, self).__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim 
        self.minibatch_size = minibatch_size
        self.num_layers = num_layers
        self.encoder = nn.Embedding(embedding_dim, hidden_dim)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers)
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def init_hidden(self, batch_size, cuda=False):
        if cuda == False:
            return (Variable(torch.zeros(self.num_layers, batch_size, 
                                                             self.hidden_dim)),
                    Variable(torch.zeros(self.num_layers, batch_size, 
                                                             self.hidden_dim)))
        else:
            return (Variable(torch.zeros(self.num_layers, batch_size, 
                                         self.hidden_dim).cuda()),
                    Variable(torch.zeros(self.num_layers, batch_size, 
                                         self.hidden_dim)).cuda())
    def forward(self, embeds, hidden):
        batch_size = embeds.size(0)
        encoded = self.encoder(embeds)
        lstm_out, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        lstm_out = self.decoder(lstm_out.view(batch_size, -1))
        return lstm_out, hidden

    def forward2(self, embeds, hidden):
        encoded = self.encoder(embeds.view(1, -1))
        lstm_out, hidden = self.rnn(encoded.view(1, 1, -1), hidden)
        lstm_out = self.decoder(lstm_out.view(1, -1))
        return lstm_out, hidden    