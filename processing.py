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


PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<SOS>" :SOS_TOKEN ,"<EOS>":EOS_TOKEN ,
        "<PAD>":PAD_TOKEN,  "<UNK>" :UNK_TOKEN  }
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


# MAX_LENGTH = 10
# MAX_LENGTH = 100

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

def filterPair(p , max_len = 10 ):
     return len(p[0].split(' ')) < max_len and \
             len(p[1].split(' ')) < max_len
def filterPairs(pairs,max_len = 10 ):
    return [pair for pair in pairs if filterPair(pair, max_len = max_len)]

    # src_lang, tgt_lang, pairs = readLangs(lang1, lang2, reverse)
    # print("Read %s sentence pairs" % len(pairs))
    # pairs = filterPairs(pairs)
    # print("Trimmed to %s sentence pairs" % len(pairs))
    # print("Counting words...")
    # for pair in pairs:
    #     src_lang.addSentence(pair[0])
    #     tgt_lang.addSentence(pair[1])
    # print("Counted words:")
    # print(src_lang.name, src_lang.n_words)
    # print(tgt_lang.name, tgt_lang.n_words)
    # return src_lang, tgt_lang, pairs



def prepareData(lang1, lang2, reverse=False , max_len = 10):
    src_lang, tgt_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs, max_len = max_len)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        src_lang.addSentence(pair[0])
        tgt_lang.addSentence(pair[1])
    src_lang.get_index2word()
    tgt_lang.get_index2word()
    print("Number of unique words:")
    print(src_lang.name, src_lang.n_words)
    print(tgt_lang.name, tgt_lang.n_words)
    return src_lang, tgt_lang, pairs


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def phrase_to_tensor(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    # indexes.append(EOS_TOKEN)
    # indexes = [SOS_TOKEN] + indexes
    #before modification
    # return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)
    return torch.tensor(indexes, dtype=torch.long, device=device)
    return indexes

def pair_to_tensor(pair):
    input_arr = phrase_to_tensor(src_lang, pair[0])
    target_arr = phrase_to_tensor(tgt_lang, pair[1])
    return (input_arr, target_arr)




class BatchDataset():
    def __init__(self,src_lang,tgt_lang,
                 text_pairs):
        """
        Creates a batch dataset (sorting sentences by size)
        and padding sentences to have the max size.
        """

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        raw_data = [ pair_to_tensor(t) for t in text_pairs]
        src = [ a[0] for a in raw_data]
        tgt = [ a[1] for a in raw_data] 

        #len of sentences
        src_lens = np.array([len(a) for a in src])
        tgt_lens = np.array([len(a) for a in tgt])


        # PROBLEMO HERE AS SORTING ONE BY THE OTHER
        #sort sentences by size
        ord_src = src_lens.argsort()
        src = [src[ind] for ind in ord_src]
        tgt = [tgt[ind] for ind in ord_src]

        #recalculate lens
        self.src_lens = [len(a) for a in src]
        self.tgt_lens = [len(a) for a in tgt]

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


    def batch_data(self,bs=100):
        """
        Batches the data padding sentences to the size
        of the maximum phrase in the batch. 
        SOS and EOS are added to the target,
        whilst just EOS is added to the source.
        """
        no_batches = len(self.src_lens) // bs
        excess = len(self.src_lens) % bs 
        batches = []
        for bid in range(no_batches):
            #normal batch
            if bid!= no_batches - 1 :
                #max sizes for padding sentences
                src_max_len = max(self.src_lens[bs*bid:bs*(bid+1)])
                tgt_max_len = max(self.tgt_lens[bs*bid:bs*(bid+1)])

                #src sentences have length = max +  1 (for EOS)
                src_batch = torch.ones((no_batches,src_max_len+1))* PAD_TOKEN
                #tgt sentences have length = max + 1 (for SOS and EOS)
                tgt_batch = torch.ones((no_batches,tgt_max_len+2))* PAD_TOKEN

                #set SOS token for tgt batch
                tgt_batch[:,0] = SOS_TOKEN

  
                for pid in range(bs):
                    
                    #find position of src_phrase in contigous array
                    src_start,src_end = self.src_pos[bid*bs+pid : bid*bs+pid+2]
                    #extract tensor of sentence
                    phrase_src = self.src[src_start:src_end]
                    pl = len(phrase_src)
                    #insert values into the batch tensor and add EOS
                    src_batch[pid,:pl] = phrase_src
                    src_batch[pid,pl] = EOS_TOKEN

                    #find position of tgt phrase in contiguous array
                    tgt_start,tgt_end = self.tgt_pos[bid*bs+pid : bid*bs+pid+2]
                    #extract tensor of sentence
                    phrase_tgt = self.tgt[tgt_start:tgt_end]
                    pl = len(phrase_tgt)
                    #insert values into the batch tensor and add EOS
                    tgt_batch[pid,1:pl+1] = phrase_tgt
                    tgt_batch[pid,pl+1] = EOS_TOKEN

                import pdb; pdb.set_trace()
            #final batch will be larger as it contains excess phrases
            else:
                pass





if __name__=="__main__":

    src_lang, tgt_lang, pairs = prepareData('eng', 'fra', True)
    dat = BatchDataset(src_lang,tgt_lang,pairs)
    dat.batch_data()


#class StreamData():
#    def __init__(self,src_lang,tgt_lang,
#                 text_pairs,seq_size=10,
#                tokens_per_batch = 100,batch_size = -1):
#        """
#        Creates a streaming data set (sentences are combined into
#        contiguous stream. Batching results in phrases of size
#        seq_size with the user controlling the number of phrases per
#        batch by either using a fixed batch size or by generating batch
#        sizes for a user selected value for the number of tokens in a batch
#        """

#        self.src_lang = src_lang
#        self.tgt_lang = tgt_lang
#        raw_data = [ pair_to_arrays(t) for t in text_pairs]
#        src =  [ a[0] for a in raw_data]
#        tgt = [ a[1] for a in raw_data]

#        #positions of sentences
#        src_pos = np.cumsum([len(a) for a in src])
#        self.src_pos = [0] + src_pos.tolist()
#        tgt_pos= np.cumsum([len(a) for a in tgt])
#        self.tgt_pos = [0]+tgt_pos.tolist()

#        self.src = torch.hstack(src)
#        self.tgt = torch.hstack(tgt)


#    def src2phrase(self,x):
#        return [self.src_lang.index2word[i] for i in x.tolist()]
#    def tgt2phrase(self,x):
#        return [self.tgt_lang.index2word[i] for i in x.tolist()]


#    def batch_data(self,seq_size,tokens_per_batch,batch_size):
#        assert batch_size ==-1 or tokens_per_batch == -1
#        assert not (batch_size == -1 and tokens_per_batch == -1)

#        if batch_size == -1:
#            #batch by tokens per batch
#            pass
#        elif tokens_per_batch == -1:
#            excess_ind_src = len(self.src) % seq_size
#            excess_ind_tgt = len(self.tgt) % seq_size