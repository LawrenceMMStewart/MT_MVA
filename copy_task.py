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

def create_dataset(all_batches):
    for src,tgt in all_batches:
        yield Batch(src, tgt, 0)

def identity_gen(V, batch, nbatches,phrase_size = 10):
    """Generates data sampled from the identity function"""
    all_batches = []
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 
            phrase_size + 1)))
        data[:, 0] = 1

        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        all_batches.append((src,tgt))
        # yield Batch(src, tgt, 0)
    return all_batches


def identity_varied_size_gen(V, batch, nbatches,phrase_size = 10):
    """Generates data sampled from the identity function
    with varied size of phrases (to test padding works)"""
    all_batches = []
    for i in range(nbatches):
        data =np.random.randint(1, V, size=(batch, 
            phrase_size + 1))
        data[:, 0] = 1

        #cut sentences to random sizes
        inds = np.random.randint(1,data.shape[1]+1,(data.shape[0],1))
        inds = inds.repeat(data.shape[1],axis=1)
        pos = np.vstack([np.arange(0,inds.shape[1])]*inds.shape[0])
        inds = np.where(pos>inds)

        
        data[inds] = 0 #change some things to padding 
        data = torch.from_numpy(data)

        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        all_batches.append((src,tgt))
        # yield Batch(src, tgt, 0)
    return all_batches

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()

        #calculate acc
        acc = x.contiguous().view(-1, x.size(-1)).argmax(dim=1) == y.contiguous().view(-1) 
        acc = acc.sum().item()
        return loss.item() * norm, acc



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



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Toy tasks')
    parser.add_argument('--V', default = 10 , type = int,
                    help='Size of the vocab to copy')
    parser.add_argument('--phrase_size',default = 30, type = int , help = 'size of sentences')
    parser.add_argument('--bs',default = 30 ,type = int, 
        help = 'batch size - no. sentences per batch')
    parser.add_argument('--train_size',default = 50, type = int , 
        help = 'number of batches in the train dataset')
    parser.add_argument('--eval_size',default = 30, type = int , 
        help = 'number of batches in the evaluation dataset')
    parser.add_argument('--no_epochs',default = 50, type =int,
        help = 'number of epochs for training')
    parser.add_argument('--warmup',default = 400, type = int,
        help = 'number of iterations till max point of lr ')
    parser.add_argument('--task',default = 'id',type=str,
        help = 'task to solve ops : id , id_varied ')
    args = parser.parse_args()





    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    V = args.V + 1
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0) #no smoothing here
    model = make_model(V, V, N=2).to(device)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, args.warmup,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)) #was 400 before

    elapsed_time = 0
    tlosses = []
    elosses = []
    taccs = []
    eacss = []

    if args.task == 'id':
        data_gen = identity_gen
    elif args.task == 'id_varied':
        data_gen = identity_varied_size_gen
    else:
        raise Exception("Please select a valid task")

    
    print(f'--Running {args.task}')

    train_batches  = data_gen(V, args.bs, 
        args.train_size , phrase_size = args.phrase_size)
    eval_batches  = data_gen(V, args.bs, args.eval_size,
        phrase_size = args.phrase_size)


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
