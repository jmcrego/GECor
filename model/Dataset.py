# -*- coding: utf-8 -*-

import sys
import os
import torch
from tqdm import tqdm
import logging
import numpy as np
from collections import defaultdict
from model.Tokenizer import Tokenizer
from model.Vocab import Vocab
from model.Utils import create_logger

separ = '￨'

def debug_batch(idxs, batch_ids, batch_ids2words, batch_reftags, batch_refwords, tokenizer):
    print('Batch {}'.format(idxs))
    ids = batch_ids.tolist()
    i2w = batch_ids2words.tolist()
    tag = batch_reftags.tolist()
    wrd = batch_refwords.tolist()
    for k in range(len(idxs)):
        print("{} txt [{}]: {}".format(idxs[k], len(ids[k]), [tokenizer.get_str([_],False) for _ in ids[k]]))
        print("{} ids [{}]: {}".format(idxs[k], len(ids[k]), ids[k]))
        print("{} i2w [{}]: {}".format(idxs[k], len(i2w[k]), i2w[k]))
        print("{} tag [{}]: {}".format(idxs[k], len(tag[k]), tag[k]))
        print("{} wrd [{}]: {}".format(idxs[k], len(wrd[k]), wrd[k]))

def pad_listoflists(ll, pad=0, maxl=0):
    if not maxl:
        maxl = max([len(l) for l in ll])
    for i in range(len(ll)):
        if len(ll[i]) > maxl:
            logging.error('Bad input data ll: {}'.format(ll))
            sys.exit()
        if pad >= 0:
            ll[i] += [pad] * (maxl-len(ll[i]))
        else:
            while len(ll[i]) < maxl:
                ll[i].append(ll[i][-1]+1)
            
    return torch.Tensor(ll).to(dtype=torch.long) ### convert to tensor
            
class Dataset():

    def __init__(self, fname, tags, words, tokenizer, args):
        super(Dataset, self).__init__()
        if not os.path.isfile(fname):
            logging.error('Cannot read file {}'.format(fname))
            sys.exit()
        self.tags = tags
        self.words = words
        self.tokenizer = tokenizer
        self.args = args
        self.idx_PAD_tgt = words.idx_PAD ### pad in tgt vocabularies (tags and words)
        self.PAD_tgt = words.PAD ### pad in tgt vocabularies (tags and words)
        self.idx_BOS_src = 0 ### <s> in src vocabulary (encoder)
        self.idx_EOS_src = 1 ### </s> in src vocabulary (encoder)
        self.idx_PAD_src = 2 ### <pad> in src vocabulary (encoder)
        self.is_inference = False
        self.Data = []

        with open(fname, 'r') as fd:
            n_filtered = 0
            for idx, l in enumerate(fd):
                toks = l.rstrip().split('\t')
                #example_with_n_predictions = 0
                if len(toks) == 2: ### training
                    words = ['<s>']
                    tags = [self.PAD_tgt]
                    for wordtag in toks[0].split(): #list of strings
                        word, tag = wordtag.split(separ)
                        words.append(word)
                        tags.append(tag)
                        #example_with_n_predictions += 1
                    assert(len(words) == len(tags))
                    lids = eval(toks[1])
                    ids2words = [0] #corresponds to <s>
                    ids = [self.idx_BOS_src]
                    for i_word in range(len(lids)):
                        for i in range(len(lids[i_word])):
                            ids2words.append(i_word+1)
                            ids.append(lids[i_word][i])

                    words.append('</s>')
                    tags.append(self.PAD_tgt)
                    ids.append(self.idx_EOS_src)
                    ids2words.append(ids2words[-1]+1)
                    assert(len(ids) == len(ids2words))
                    # words    : <s>   This is  my  exxample       </s>
                    # ids      : 0     234  31  67  35 6789        1
                    # ids2words: 0     1    2   3   4  4           5
                    # tags     : <PAD> ·    ·   ·   $SPELL_example <PAD>
                elif len(toks) == 1: ### inference (input is raw text)
                    self.is_inference = True
                    ids = self.tokenizer.get_ids(l.rstrip(), add_special_tokens=True)
                    words, ids2words, _ = self.tokenizer.get_words_ids2words_subwords(ids)
                    tags = []
                    
                if not self.is_inference and self.args.max_length > 0 and len(ids) > self.args.max_length:
                    n_filtered += 1
                    continue

                self.Data.append({'words':words , 'tags':tags, 'ids':ids, 'ids2words':ids2words, 'idx':idx})

        logging.info('Read [{}] dataset={} with {} examples [{} filtered]'.format('inference' if self.is_inference else 'learning', fname,len(self.Data),n_filtered))
        
    def __iter__(self):
        assert len(self.Data) > 0, 'Empty dataset'
        idx_Data = [i for i in range(len(self.Data))]
        np.random.shuffle(idx_Data)
        logging.info('Shuffled dataset')
        self.args.shard_size = self.args.shard_size or len(idx_Data)
        shards = [idx_Data[i:i+self.args.shard_size] for i in range(0, len(idx_Data), self.args.shard_size)] # split dataset in shards
        logging.info('Built {} shards with up to {} examples'.format(len(shards),self.args.shard_size))
        for s,shard in enumerate(shards):
            batchs = self.build_batchs(shard)
            logging.info('Shard {}/{} contains {} batchs'.format(s+1,len(shards),len(batchs)))
            for batch in batchs:
                yield self.format_batch(batch)
            logging.info('End of shard {}/{}'.format(s+1,len(shards)))
        logging.info('End of dataset')

        
    def build_batchs(self, shard):
        shard_len = [len(self.Data[idx]['ids']) for idx in shard]
        shard = np.asarray(shard)
        ord_lens = np.argsort(shard_len) #sort by lens (lower to higher lengths)
        shard = shard[ord_lens] #examples in shard are now sorted by lens
        batchs = [] ### build batchs of same (similar) size
        curr_batch = []
        curr_batch_len = 0
        for idx in shard:
            if curr_batch_len + self.len_example(idx) > self.args.batch_size:
                if curr_batch_len:
                    batchs.append(curr_batch)
                curr_batch = []
                curr_batch_len = 0
            curr_batch.append(idx)
            curr_batch_len += self.len_example(idx)
        if curr_batch_len:
            batchs.append(curr_batch)
        np.random.shuffle(batchs)
        return batchs

    
    def len_example(self, idx):
        if self.args.batch_type == 'tokens':
            return len(self.Data[idx]['ids']) ### number of subwords
        return 1 ### number of sentences


    def format_batch(self, idxs):
        batch_words = []
        batch_ids = []
        batch_ids2words = []
        batch_reftags = []
        batch_refwords = []
        maxl_ids = 0
        for idx in idxs:
            curr_words = self.Data[idx]['words'] ### contains <s> and </s>
            curr_ids = self.Data[idx]['ids'] ### contains <s> and </s>
            maxl_ids = len(curr_ids) if len(curr_ids) > maxl_ids else maxl_ids
            curr_ids2words = self.Data[idx]['ids2words']
            curr_tags = self.Data[idx]['tags'] ### contains PAD's in the positions of <s> and </s>
            assert(len(curr_ids) == len(curr_ids2words))
            #assert(len(curr_ids) >= len(curr_tags))
            if len(curr_ids) < len(curr_tags):
                logging.info('ATTENTION [len(curr_ids) < len(curr_tags)]\nidx: {}\ncurr_ids : {}\ncurr_tags: {}'.format(idx,curr_ids,curr_tags))
                continue
            ### this part is a bit tricky... i have:
            # words    : <s>   This is  my  exxample       </s>
            # ids      : 0     234  31  67  35 6789        1
            # ids2words: 0     1    2   3   4  4           5
            # tags     : PAD   ·    ·   ·   $SPELL_example PAD
            ### i want the references (idx's) for tags/words to predict:
            # reftags  : PAD   ·    ·   ·   $SPELL         PAD
            # refwords : PAD   PAD  PAD PAD example        PAD
            curr_reftags = []
            curr_refwords = []
            if not self.is_inference:
                for i in range(len(curr_tags)): #PAD ·    ·   ·   $SPELL_example PAD
                    if '_' in curr_tags[i]: #$SPELL_example
                        tag, word = curr_tags[i].split('_')
                        if tag == self.tags.idx_UNK:
                            logging.info('ATTENTION tag: {} in {}'.format(tag,curr_tags[i]))
                        curr_reftags.append(self.tags[tag]) #idx($SPELL)
                        curr_refwords.append(self.words[word]) #idx(example)
                    else:
                        curr_reftags.append(self.tags[curr_tags[i]]) #idx(·) OR idx(PAD) OR idx(APPEND) ...
                        curr_refwords.append(self.idx_PAD_tgt) #idx(PAD)
            ### add example
            batch_words.append(curr_words)
            batch_ids.append(curr_ids)
            batch_ids2words.append(curr_ids2words)
            if not self.is_inference:
                batch_reftags.append(curr_reftags)
                batch_refwords.append(curr_refwords)
        ### convert to tensor
        batch_ids = pad_listoflists(batch_ids, pad=self.idx_PAD_src, maxl=maxl_ids)
        batch_ids2words = pad_listoflists(batch_ids2words, pad=-1, maxl=maxl_ids)
        if self.is_inference:
            return [batch_ids, batch_ids2words, batch_words, idxs]

        #if reftags/refwords are smaller than ids/ids2words i add PAD so as to obtain the same size
        batch_reftags = pad_listoflists(batch_reftags,pad=self.idx_PAD_tgt,maxl=maxl_ids)
        batch_refwords = pad_listoflists(batch_refwords,pad=self.idx_PAD_tgt,maxl=maxl_ids)
        # reftags  : PAD   ·    ·   ·   $SPELL         PAD+
        # refwords : PAD   PAD  PAD PAD example        PAD+
        if self.args.log_level == 'debug':
            debug_batch(idxs, batch_ids, batch_ids2words, batch_reftags, batch_refwords, self.tokenizer)
        return [batch_ids, batch_ids2words, batch_reftags, batch_refwords]

class Args():
    def __init__(self, batch_size = 2, batch_type = 'sentences', shard_size = 50, max_length = 100):
        self.batch_size = batch_size
        self.batch_type = batch_type
        self.shard_size = shard_size
        self.max_length = max_length
    
    
if __name__ == '__main__':

    create_logger(None,'info')
    tags = Vocab('experiment/tags')
    words = Vocab('experiment/words')
    tokenizer = Tokenizer('flaubert/flaubert_base_cased')
    args = Args()
    ds = Dataset(sys.argv[1], tags, words, tokenizer, args)
    for batch in ds:
        print('ids',batch[0])
        print('ids2words',batch[1])
        print('reftags',batch[2])
        print('refwords',batch[3])
        sys.exit()
            
