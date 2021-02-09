from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SOS_token = 0
EOS_token = 1
PAD_token = 2
UNK_token = 3

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<SOS>" :0 ,"<EOS>":1 ,"<PAD>":2,  "<UNK>" :3  }
        self.word2count = {}
        self.index2word = {} #to be created after word2index
        self.n_words = 4

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def get_index2word(self):
        self.index2word = dict([(value,key) for key,value in self.word2index.items()])

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        src_lang = Lang(lang2)
        tgt_lang = Lang(lang1)
    else:
        src_lang = Lang(lang1)
        tgt_lang = Lang(lang2)

    return src_lang, tgt_lang, pairs


MAX_LENGTH = 10

# eng_prefixes = (
#     "i am ", "i m ",
#     "he is", "he s ",
#     "she is", "she s ",
#     "you are", "you re ",
#     "we are", "we re ",
#     "they are", "they re "
# )

## in order to filter according to certain english phrases, use the above
# def filterPair(p):
#     return len(p[0].split(' ')) < MAX_LENGTH and \
#         len(p[1].split(' ')) < MAX_LENGTH and \
#         p[1].startswith(eng_prefixes)

# def filterPairs(pairs):
#     return [pair for pair in pairs if filterPair(pair)]

def filterPair(p):
     return len(p[0].split(' ')) < MAX_LENGTH and \
             len(p[1].split(' ')) < MAX_LENGTH
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

    src_lang, tgt_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        src_lang.addSentence(pair[0])
        tgt_lang.addSentence(pair[1])
    print("Counted words:")
    print(src_lang.name, src_lang.n_words)
    print(tgt_lang.name, tgt_lang.n_words)
    return src_lang, tgt_lang, pairs



def prepareData(lang1, lang2, reverse=False):
    src_lang, tgt_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        src_lang.addSentence(pair[0])
        tgt_lang.addSentence(pair[1])
    src_lang.get_index2word()
    tgt_lang.get_index2word()
    print("Counted words:")
    print(src_lang.name, src_lang.n_words)
    print(tgt_lang.name, tgt_lang.n_words)
    return src_lang, tgt_lang, pairs


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    #before modification
    # return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)
    return torch.tensor(indexes, dtype=torch.long, device=device)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(src_lang, pair[0])
    target_tensor = tensorFromSentence(tgt_lang, pair[1])
    return (input_tensor, target_tensor)

class StreamData():
    def __init__(self,src_lang,tgt_lang,
                 text_pairs,seq_size=10,
                tokens_per_batch = 100,batch_size = -1):
        """
        Creates a streaming data set (sentences are combined into
        contiguous stream. Batching results in phrases of size
        seq_size with the user controlling the number of phrases per
        batch by either using a fixed batch size or by generating batch
        sizes for a user selected value for the number of tokens in a batch
        """

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        raw_data = [ tensorsFromPair(t) for t in text_pairs]
        src =  [ a[0] for a in raw_data]
        tgt = [ a[1] for a in raw_data]

        #positions of sentences
        src_pos = np.cumsum([len(a) for a in src])
        self.src_pos = [0] + src_pos.tolist()
        tgt_pos= np.cumsum([len(a) for a in tgt])
        self.tgt_pos = [0]+tgt_pos.tolist()

        self.src = torch.hstack(src)
        self.tgt = torch.hstack(tgt)




    def src2phrase(self,x):
        return [self.src_lang.index2word[i] for i in x.tolist()]
    def tgt2phrase(self,x):
        return [self.tgt_lang.index2word[i] for i in x.tolist()]



src_lang, tgt_lang, pairs = prepareData('eng', 'fra', True)
dat = StreamData(src_lang,tgt_lang,pairs,12)
import pdb; pdb.set_trace()
