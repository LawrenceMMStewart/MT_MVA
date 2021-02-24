import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
from utils import *
from transformer import *
from processing import *

def create_dataset(all_batches):
    for src,tgt in all_batches:
        yield Batch(src, tgt, 0)

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Toy tasks')
    parser.add_argument("--seed",default=123,type=int,help= "seed for experiment")
    parser.add_argument('--max_size',default = 20, type = int , help = 'max size of sentences')
    parser.add_argument("--split", default = 0.1 ,type = float,
        help = "fraction of data to use as a validation set")
    parser.add_argument('--bs',default = 30 ,type = int, 
        help = 'batch size - no. sentences per batch')
    parser.add_argument('--no_epochs',default = 10, type =int,
        help = 'number of epochs for training')
    parser.add_argument("--no_units",default = 6,type=int,
        help = 'number of encoder decoder units to stack')
    parser.add_argument('--warmup',default = 4000, type = int,
        help = 'number of iterations till max point of lr ')
    args = parser.parse_args()



    np.random.seed(args.seed)
    #load data
    src_lang, tgt_lang, pairs = prepareData('eng', 'fra', 
        reverse=True,max_len = args.max_size)
    train_pairs , valid_pairs = shuffle_split(pairs,split = args.split)
    train = BatchDataset(src_lang,tgt_lang,train_pairs)
    valid = BatchDataset(src_lang,tgt_lang,valid_pairs)
    train_batches = train.batch_data()
    valid_batches =  valid.batch_data()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #IS THIS CORRECT I DONT KNOW PLEASE CHECK 
    V = tgt_lang.n_words
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0) #no smoothing here
    model = make_model(V, V, N=args.no_units).to(device)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, args.warmup,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)) #was 400 before

    elapsed_time = 0
    tlosses = []
    elosses = []
    taccs = []
    eacss = []


    for epoch in range(args.no_epochs):

        train_set =  create_dataset(train_batches)
        eval_set =  create_dataset(eval_batches)

        #Train the model
        model.train()

        train_loss ,train_acc, train_time = run_epoch(train_set, model, 
                  SimpleLossCompute(model.generator, criterion, model_opt))

        #Eval the model
        model.eval()

        eval_loss, eval_acc,eval_time = run_epoch(eval_set, model, 
                        SimpleLossCompute(model.generator, criterion, None))
        elapsed_time = elapsed_time + train_time + eval_time 

        print(f'Epoch {epoch} : TLoss = {train_loss:.4f} , Tacc= {train_acc:.2f} \
            ELoss = {eval_loss:.4f}, Eacc = {eval_acc:.2f}, \
            Elapsed Time = {elapsed_time:.1f}, \
            Lr (start of epoch) {model_opt._rate:.4f}')

        tlosses.append(train_loss)
        elosses.append(eval_loss)
        taccs.append(train_acc)
        eacss.append(eval_acc)



    plt.figure()
    plt.style.use('ggplot')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(tlosses,label = 'Train')
    plt.plot(elosses,label=  'eval')
    plt.legend()
    plt.grid('on')
    plt.savefig(f'{args.task}_losses.png')
    plt.show()



    plt.figure()
    plt.style.use('ggplot')
    plt.xlabel("Epoch")
    plt.ylabel("acc")
    plt.plot(taccs,label = 'Train')
    plt.plot(eacss,label=  'eval')
    plt.legend()
    plt.grid('on')
    plt.savefig(f'{args.task}_accs.png')
    plt.show()


    # model.eval()
    # src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]) )
    # src_mask = Variable(torch.ones(1, 1, 10) )
    # print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))
