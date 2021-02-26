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
import os 
import json

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

    parser = argparse.ArgumentParser(description='Machine Translation Training ; Fr -> Eng')
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
    parser.add_argument("--dropout", default = 0.1 ,type = float,
        help = "dropout for encoder decoder")
    parser.add_argument("--d_model", default = 512, type = int,
        help = "dimension of the transformer model")
    parser.add_argument("--no_heads", default = 8 ,type = int,
        help = "number of heads in multi-attention")
    parser.add_argument('--warmup',default = 4000, type = int,
        help = 'number of iterations till max point of lr ')
    parser.add_argument("--smoothing", default = 0.0 , type = float,
        help = "smoothing coeffecient into label smoothing (between 0-1)")
    parser.add_argument("--exp_name",default = "test",type = str,
        help = "Name of experiment, to save plots")
    args = parser.parse_args()

    np.random.seed(args.seed)

    #create output folder:
    PATH = f'results/{args.exp_name}'
    assert not os.path.isdir(PATH) ; "exp already exists"
    os.mkdir(PATH)


    #load data
    src_lang, tgt_lang, pairs = prepareData('eng', 'fra', 
        reverse=True,max_len = args.max_size)
    train_pairs , valid_pairs = shuffle_split(pairs,split = args.split)
    train = BatchDataset(src_lang,tgt_lang,train_pairs)
    valid = BatchDataset(src_lang,tgt_lang,valid_pairs)
    train_batches = train.batch_data()
    eval_batches =  valid.batch_data()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #IS THIS CORRECT I DONT KNOW PLEASE CHECK 
    V_src = src_lang.n_words
    V_tgt = tgt_lang.n_words
    criterion = LabelSmoothing(size=V_tgt, padding_idx=0, smoothing=0.0) #no smoothing here

    model = make_model(V_src, V_tgt, N=args.no_units,
        d_model = args.d_model, h = args.no_heads,
        dropout = args.dropout).to(device)

    model_opt = NoamOpt(model.src_embed[0].d_model, 1, args.warmup,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)) 

    elapsed_time = 0
    tlosses = []
    elosses = []
    taccs = []
    eacss = []

    start_time = time.time()
    for epoch in range(args.no_epochs):

        train_set =  create_dataset(train_batches)
        eval_set =  create_dataset(eval_batches)

        #Train the model
        model.train()

        train_loss ,train_acc = run_epoch(train_set, model, 
                  SimpleLossCompute(model.generator, criterion, model_opt))

        #Eval the model
        model.eval()

        eval_loss, eval_acc = run_epoch(eval_set, model, 
                        SimpleLossCompute(model.generator, criterion, None))
    
        elapsed_time = time.time() - start_time
        print(f'Epoch {epoch} : TLoss = {train_loss:.4f} , Tacc = {train_acc:.2f}, TrainPP={np.exp(train_loss):.2f}\
            ELoss = {eval_loss:.4f}, Eacc = {eval_acc:.2f}, EvalPP = {np.exp(train_loss):.2f}\
            Elapsed Time = {elapsed_time:.1f}, \
            Lr (start of epoch) {model_opt._rate:.4f}')

        tlosses.append(train_loss)
        elosses.append(eval_loss)
        taccs.append(train_acc)
        eacss.append(eval_acc)


    xvals = [i+1 for i in range(len(tlosses))]

    plt.figure()
    plt.style.use('ggplot')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(xvals,tlosses,label = 'Train')
    plt.plot(xvals,elosses,label=  'eval')
    plt.legend()
    plt.grid('on')
    plt.savefig(PATH + '/losses.png')
    plt.show()



    plt.figure()
    plt.style.use('ggplot')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(xvals,taccs,label = 'Train')
    plt.plot(xvals,eacss,label=  'eval')
    plt.legend()
    plt.grid('on')
    plt.savefig(PATH + '/accs.png')
    plt.show()

    plt.figure()
    plt.style.use('ggplot')
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.plot(xvals,np.exp(tlosses).tolist(),label = 'Train')
    plt.plot(xvals,np.exp(elosses).tolist(),label= 'eval')
    plt.legend()
    plt.grid('on')
    plt.savefig(PATH + '/pps.png')
    plt.show()

    params = vars(args)
    params['tlosses'] = tlosses
    params['taccs'] = taccs
    params['tpps'] = np.exp(tlosses).tolist()
    params['elosses'] = elosses
    params['eacss'] = eacss
    params['epps'] = np.exp(elosses).tolist()
    #save the experiment parameters and losses
    with open(PATH+'/params.json','w') as f:
        f.write(json.dumps(params))
    torch.save(model.state_dict(), PATH+'/model.pth')


    # model.eval()
    # src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]) )
    # src_mask = Variable(torch.ones(1, 1, 10) )
    # print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))
