# -*- coding: utf-8 -*-
import torch
from torch.utils.data.dataset import Dataset
import unidecode
import re
import time
import math
import string

def code2char(code,vocab):
    vocab_map = dict(zip(vocab.values(),vocab.keys()))
    return "".join(vocab_map[c] for c in code)
def char2code(text,vocab):
    data = torch.ByteTensor(len(text))
    for i,c in enumerate(text):
        data[i]=vocab[c]
    return data

all_characters = string.printable
n_characters = len(all_characters)

def read_file(filename):
    file = unidecode.unidecode(open(filename).read())
    return file, len(file)


def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = all_characters.index(string[c])
        except:
            continue
    return tensor

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def make_files(text_file,out_tensorfile=None,out_vocabfile=None,vocab_file=None):
    """ Permet de creer a partir d'un fichier texte le tenseur 1D encode en entier et le mapping
    des caracteres vers leur code.
    * text_file : le fichier texte brut
    * out_tensorfile : si specifie, sauve le tenseur 1D de sortie
    * out_vocabfile : si specifie, sauve le mapping entre caractere et code
    * vocab_file : si specifie, charge le mapping entre caractere et code
    """
    with open(text_file) as f:
            text = re.sub(r'[^a-zA-Z0-9_\s]','',f.read().lower())
            text = re.sub(r'\s+',' ',text)
    if vocab_file is not None:
        vocab = torch.load(vocab_file)
    else:
        chars = set(text)
        vocab = dict(zip(sorted(chars),range(len(chars))))
    data = char2code(text,vocab)
    if out_vocabfile is not None:
        torch.save(vocab,out_vocabfile)
    if out_tensorfile is not None:
        torch.save(data,out_tensorfile)
    return data,vocab


class CharDataset(Dataset):
    """
        Charge un fichier data_file tenseur 1D d'entiers et le decoupe en sequences de longueur seq_length. Vocab_file est un dictionnaire
        de characteres vers entier.
    """

    def __init__(self,data_file,vocab_file,seq_length):
        self.data_file = data_file
        self.vocab_file = vocab_file
        self.seq_length = seq_length
        self.data = torch.load(data_file)
        self.vocab = torch.load(vocab_file)
        self.vocab_map = dict(zip(self.vocab.values(),self.vocab.keys()))
        self.nb_samples = self.data.size()[0]//self.seq_length
        print('cutting off end of data so that the batches/sequences divide evenly')
        self.data = self.data[:(self.seq_length*self.nb_samples+1)]
        self.vocab_size = len(self.vocab)
    def __getitem__(self,index):
        start,end = index*self.seq_length,(index+1)*self.seq_length
        return self.data[start:end],self.data[(start+1):(end+1)]
    def __len__(self):
        return self.nb_samples

