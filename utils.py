import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy
from torch.autograd import Variable
import time
import numpy as np



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum') #CHANGED FROM size_average = None
        # self.criterion2 = nn.NLLLoss(reduction = 'sum') #equivalent to using that
        #we use KLDivLoss over NLLLoss as we want to mask any padding tokens
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size # number of classes
        self.true_dist = None
        
    def forward(self, x, target):

        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))



def run_epoch(data_iter, model, loss_compute,desc = ""):
    """
    runs one training epoch of a dataset

    inputs:
    data_iter : iterable
    model : nn.model
    loss_compute : func 

    loss_compute takes in inputs out,batch_trg_y batch.ntokens
    where out is a tensor, batch_trg_y is a tensor 
    and batch.ntokens is an integer.
    """
    # start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    total_acc = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src.to(device), batch.trg.to(device), 
                            batch.src_mask.to(device), batch.trg_mask.to(device))
        
        loss , acc= loss_compute(out, batch.trg_y.to(device), batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        total_acc += acc
        # if i % 50 == 0:
        #     # elapsed = time.time() - start
        #     # print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
        #     #         (i, loss / batch.ntokens, tokens / elapsed))
        #     start = time.time()
        #     tokens = 0
    # elapsed_time = time.time() - start
    epoch_loss = total_loss / total_tokens
    epoch_acc = total_acc / total_tokens

    return epoch_loss.item() , epoch_acc.item()


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




# global max_src_in_batch, max_tgt_in_batch
# def batch_size_fn(new, count, sofar):
#     "Keep augmenting batch and calculate total number of tokens + padding."
#     global max_src_in_batch, max_tgt_in_batch
#     if count == 1:
#         max_src_in_batch = 0
#         max_tgt_in_batch = 0
#     max_src_in_batch = max(max_src_in_batch,  len(new.src))
#     max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
#     src_elements = count * max_src_in_batch
#     tgt_elements = count * max_tgt_in_batch
#     return max(src_elements, tgt_elements)
