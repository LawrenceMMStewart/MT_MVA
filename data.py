from torchtext import data, datasets
import spacy

spacy_en = spacy.load('en')
spacy_es = spacy.load('de')

def tokenize_en(text):
	return [tok.text for tok in spacy_en.tokenizer(text)]
def tokenize_fr(text):
	return [tok.text for tok in spacy_fr.tokenizer(text)]
def tokenize_de(text):
	return [tok.text for tok in spacy_de.tokenizer(text)]
def tokenize_es(text):
	return [tok.text for tok in spacy_es.tokenizer(text)]


BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"
MAX_LEN = 100
MIN_FREQ = 2


SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
TGT = data.Field(tokenize=tokenize_en, init_token = BOS_WORD, 
                 eos_token = EOS_WORD, pad_token=BLANK_WORD)

train, val, test = datasets.IWSLT.splits(
    exts=('.es', '.en'), fields=(SRC, TGT), 
    filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
        len(vars(x)['trg']) <= MAX_LEN)

SRC.build_vocab(train.src, min_freq=MIN_FREQ)
TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

import pdb ; pdb.set_trace()