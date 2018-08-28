# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
#from torch.utils.data import DataLoader
import time
import random
import argparse
import os
from model import LSTMt
#from charDataset import CharDataset
import string
from charDataset import char_tensor
from charDataset import time_since
from charDataset import read_file
from generate import generate

def random_training_set(file, file_len, chunk_len, batch_size):
    inp = torch.LongTensor(batch_size, chunk_len)
    target = torch.LongTensor(batch_size, chunk_len)
    for bi in range(batch_size):
        start_index = random.randint(0, file_len - chunk_len)
        end_index = start_index + chunk_len + 1
        chunk = file[start_index:end_index]
        inp[bi] = char_tensor(chunk[:-1])
        target[bi] = char_tensor(chunk[1:])
    inp = Variable(inp)
    target = Variable(target)
    if args.cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target

def train(embed, target):
    if args.cuda:
        hidden = decoder.init_hidden(args.batch_size, cuda=True)
    else:
        hidden = decoder.init_hidden(args.batch_size)
    decoder.zero_grad()
    loss = 0
    for c in range(args.chunk_len):
        output, hidden = decoder(embed[:, c], hidden)
        loss += criterion(output.view(args.batch_size, -1), target[:, c])
    loss.backward()
    decoder_optimizer.step()
    return loss.data[0] / args.chunk_len

def save():
    save_filename = os.path.splitext(os.path.basename(args.filename))[0] + \
                                                                        '.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)

#def read_file(filename, vocabname):
#    cdset = CharDataset(filename,vocabname,10)
#    return cdset, len(cdset)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('filename', type=str)#,default="train_data.tx")
    argparser.add_argument('--model', type=str, default="gru")
    argparser.add_argument('--n_epochs', type=int, default=2000)
    argparser.add_argument('--print_every', type=int, default=25)
    argparser.add_argument('--hidden_size', type=int, default=100)
    argparser.add_argument('--n_layers', type=int, default=2)
    argparser.add_argument('--learning_rate', type=float, default=0.01)
    argparser.add_argument('--chunk_len', type=int, default=200)
    argparser.add_argument('--batch_size', type=int, default=100)
    argparser.add_argument('--cuda', action='store_true')
    args = argparser.parse_args()

    if args.cuda:
	    print("CUDA activé")

    all_characters = string.printable
    n_characters = len(all_characters)
    cdset, cdset_len = read_file(args.filename)#, "vocab.tx")#)
    #dataload = DataLoader(cdset,batch_size=1,shuffle=True)
    #nb_lettres = len(torch.load("vocab.tx"))
    embedding_dim =  n_characters
#    embedding_dim = nb_lettres

	# Initialisation du modèle et training
    decoder = LSTMt(embedding_dim, args.hidden_size, embedding_dim, 
                                            args.n_layers, args.batch_size)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), 
                                                     lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    if args.cuda:
	    decoder.cuda()

    start = time.time()
    all_losses = []
    loss_avg = 0

    try:
	    print("Train de %d epochs" % args.n_epochs)
	    for epoch in range(1, args.n_epochs + 1):
	        loss = train(*random_training_set(cdset, cdset_len, args.chunk_len,
                                                              args.batch_size))
	        loss_avg += loss

	        if epoch % args.print_every == 0:
	            print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch 
                                                  / args.n_epochs * 100, loss))
	            print(generate(decoder, 'Wh', 100, cuda=args.cuda)
                                                                        , '\n')

	    print("Sauvegarde")
	    save()

    except KeyboardInterrupt:
	    print("Sauvegarde avant de quitter")
	    save()
