# -*- coding: utf-8 -*-

import sys
import os
#from unittest import result
import torch
#from tqdm import tqdm
import json
import logging
import numpy as np
from collections import defaultdict
from model.Vocab import Vocab
from utils.Utils import create_logger, SEPAR1, SEPAR2, KEEP

def debug_batch(idxs, batch_raw, batch_ids_src, batch_ids_agg, batch_ids_tag, batch_ids_cor):
    logging.debug('Batch {}'.format(idxs))
    src = batch_ids_src.tolist()
    agg = batch_ids_agg.tolist()
    tag = batch_ids_tag.tolist()
    cor = batch_ids_cor.tolist()
    for k in range(len(idxs)):
        logging.debug("{} raw: {}".format(idxs[k], batch_raw[k]))
        logging.debug("{} src [{}]: {}".format(idxs[k], len(src[k]), src[k]))
        logging.debug("{} agg [{}]: {}".format(idxs[k], len(agg[k]), agg[k]))
        logging.debug("{} tag [{}]: {}".format(idxs[k], len(tag[k]), tag[k]))
        logging.debug("{} cor [{}]: {}".format(idxs[k], len(cor[k]), cor[k]))

def pad_listoflists(ll, pad=0, maxl=0, n_subtokens=1):
    #logging.info('maxl: {}'.format(maxl))
    if maxl==0:
        maxl = max([len(l) for l in ll])
    #logging.info('maxl: {}'.format(maxl))
    for i in range(len(ll)):
        if len(ll[i]) > maxl:
            #logging.error('Bad input data ll: {}'.format(ll))
            sys.exit()
        if isinstance(pad, int) and pad == -1: ### use ll[i][-1]+1 for indexs of coo
            while len(ll[i]) < maxl:
                ll[i].append(ll[i][-1]+1)
        else: ### fill the remaining tokens using pad
            ll[i] += [pad] * (maxl-len(ll[i]))
        #logging.info('ll[{}]: {}'.format(i,ll[i]))
    return torch.Tensor(ll).to(dtype=torch.long) ### convert to tensor
            
class Dataset():

    def __init__(self, fname, tags, cors, token, args):
        super(Dataset, self).__init__()
        if not os.path.isfile(fname):
            logging.error('Cannot read file {}'.format(fname))
            sys.exit()
        self.tags = tags
        self.cors = cors #may be None
        self.token = token
        #self.noiser = noiser
        self.args = args
        #
        self.idx_BOS_src = token.idx_BOS # <s> in encoder
        self.idx_EOS_src = token.idx_EOS # </s> in encoder
        self.idx_PAD_src = token.idx_PAD # <pad> in encoder
        #
        self.str_BOS_src = token.str_BOS # <s> in encoder
        self.str_EOS_src = token.str_EOS # </s> in encoder
        self.str_PAD_src = token.str_PAD # <pad> in encoder
        #
        self.idx_PAD_tag = tags.idx_PAD # <pad> in tag vocabulary
        #
        self.idx_PAD_cor = cors.idx_PAD if cors is not None else token.idx_PAD # <pad> in cor vocabulary 
        self.Data = []
        with open(fname, 'r') as fd:
            n_filtered = 0
            for idx, l in enumerate(fd):
                ldict = json.loads(l)
                ldict.insert(0,{'r':self.str_BOS_src, 's':'s', 'l':'', 'i':[self.idx_BOS_src], 'T':self.idx_PAD_tag, 'C':self.idx_PAD_cor})
                ldict.append(  {'r':self.str_EOS_src, 's':'s', 'l':'', 'i':[self.idx_EOS_src], 'T':self.idx_PAD_tag, 'C':self.idx_PAD_cor})
                #ldict = noiser.noise(json.loads(l)) #list of words (dicts) with noise injected
                #print(ldict)
                self.Data.append({'idx':idx, 'ldict':ldict})
        logging.info('Read {} examples from {}'.format(len(self.Data),fname))
        sys.exit()
        
        self.n_truncated = 0
        self.n_filtered = 0
        self.n_remaining = 0
        for idx in range(len(self.Data)):
            if 'ids_src' in self.Data[idx]: ### already parsed
                continue
            tokens = self.Data[idx]['raw'].split() ### data MUST be already splitted into tokens, ex: #Son￤593￤$SWAP ... rendu￤1556￤$PHONE￤rendue￤4072 
            if self.args.max_length > 0 and len(tokens) > self.args.max_length:
                self.n_truncated += 1
                tokens = tokens[:self.args.max_length]
            if len(tokens) == 0:
                self.n_filtered += 1
                self.Data[idx]['ids_src'] = []
                logging.warning('filtered {} line'.format(idx))
                continue
            self.n_remaining += 1
            ids_src, ids_tag, ids_cor, ids_agg, str_src = self.format_sentence(tokens)
            self.Data[idx]['ids_src'] = ids_src
            self.Data[idx]['ids_tag'] = ids_tag
            self.Data[idx]['ids_cor'] = ids_cor
            self.Data[idx]['ids_agg'] = ids_agg
            self.Data[idx]['str_src'] = str_src
        logging.info('Parsed dataset [filtered={}, truncated={}, remain={}]'.format(self.n_filtered, self.n_truncated, self.n_remaining))

    def format_sentence(self, tokens):
        #
        ### AGGREGATE (sum,avg,max)
        str_src = [] # <s>   This  is    my    exxample  </s>
        ids_src = [] # 0     234   31    67    35  678   1     (0:<s>, 1:</s>)
        ids_agg = [] # 0     1     2     3     4   4     5     (indicates to which tok is aligned each subtoken)
        ids_tag = [] # 0     1     1     1     4         0     (0:<PAD>, 1:$KEEP, 4:$SPELL)
        ids_cor = [] # [0,0] [0,0] [0,0] [0,0] [3245,4]  [0,0] (0:<PAD>, 3245:example, when n_subtokens=2)
        #
        ### GECTOR (first,last)
        str_src = [] # <s>   This  is    my    exxxample   </s>
        ids_src = [] # 0     234   31    67    35    678   1     (0:<s>, 1:</s>)
        ids_agg = [] # 0     1     2     3     4     4     5     (indicates to which tok is aligned each subtoken) ### used for inference
        ids_tag = [] # 0     1     1     1     4     0     0     (0:<PAD>, 1:$KEEP, 4:$SPELL)
        ids_cor = [] # [0,0] [0,0] [0,0] [0,0] [7,0] [0,0] [0,0] (0:<PAD>, 3245:example, when n_subtokens=2)
        #
        str_src.append('<s>')
        ids_src.append(self.idx_BOS_src)
        ids_agg.append(0)
        ids_tag.append(self.idx_PAD_tag)
        ids_cor.append([self.idx_PAD_cor]*self.args.n_subtokens)
        for i,tok in enumerate(tokens):
            fields = tok.split(SEPAR1)
            #7 fields: Suivent￤12581～1558￤Aa￤True￤$KEEP￤Suivent￤26743
            #5 fields: versets￤35110￤a￤True￤$KEEP
            n_subtok = len(fields[1].split(SEPAR2))
            ids_agg += [i+1]*n_subtok #ex: [1,1]
            str_src += [fields[0]] #Suivent
            ids_src += self.build_ids_src(fields[1]) #[12581,1558]
            assert(len(ids_src) == len(ids_agg))
            if self.args.aggregation in ['max', 'avg', 'sum']:
                ids_tag += self.build_tag_aggregate(fields[4])
                ids_cor += self.build_cor_aggregate(fields[6]) if len(fields) == 7 else [[self.idx_PAD_cor]*self.args.n_subtokens]
                assert(len(ids_cor) == len(ids_tag))
            elif self.args.aggregation in ['first', 'last']: ### gector-like
                ids_tag += self.build_tag_gector(fields[4],n_subtok)
                ids_cor += self.build_cor_gector(fields,n_subtok)
                assert(len(ids_src) == len(ids_tag))
                assert(len(ids_src) == len(ids_cor))
            else:
                logging.error('Bad aggregation option')
                sys.exit()                
        str_src.append('</s>')
        ids_src.append(self.idx_EOS_src)
        ids_tag.append(self.idx_PAD_tag)
        ids_cor.append([self.idx_PAD_cor]*self.args.n_subtokens)
        ids_agg.append(ids_agg[-1]+1)
        return ids_src, ids_tag, ids_cor, ids_agg, str_src

    def build_ids_src(self,field_ids):
        return list(map(int,field_ids.split(SEPAR2)))

    def build_tag_aggregate(self, field_tag):
        return [self.tags[field_tag]]
        
    def build_cor_aggregate(self, field_cor):
        myids_cor = list(map(int,field_cor.split(SEPAR2)))
        if len(myids_cor) < self.args.n_subtokens:
            myids_cor += [4]*(self.args.n_subtokens-len(myids_cor)) #4 corresponds to flaubert model token: <special0> that must be predicted (PAD is not)
        elif len(myids_cor) > self.args.n_subtokens:
            myids_cor = myids_cor[:self.args.n_subtokens] #truncation... problem if the right word (using n subtokens) cannot be produced
        return [myids_cor]
            
    def build_tag_gector(self,field_tag,l):
        myids_tag = self.tags[field_tag]
        ltag = [self.idx_PAD_tag] * l
        if self.args.aggregation == 'first':
            ltag[0] = myids_tag
        else:
            ltag[-1] = myids_tag
        return ltag
        
    def build_cor_gector(self,fields,n_subtok):
        ### n_subtok is the number of subtokens of current source words
        ### self.args.n_subtokens is the number of subtokens to be predicted as correction
        if len(fields)<7:
            myids_cor = [self.idx_PAD_cor]*self.args.n_subtokens
        else:
            myids_cor = list(map(int,fields[6].split(SEPAR2)))
            
        if len(myids_cor) < self.args.n_subtokens:
            myids_cor += [4]*(self.args.n_subtokens-len(myids_cor)) #4 corresponds to flaubert model token: <special0> that must be predicted (PAD is not)
        elif len(myids_cor) > self.args.n_subtokens:
            myids_cor = myids_cor[:self.args.n_subtokens]

        pad_cor = [self.idx_PAD_cor] * self.args.n_subtokens ### a word correction is formed of n_subtoken <pad>'s
        lcor = [pad_cor] * n_subtok ### initially all subtokens of current word are padded
        ### exx   ample
        ### [0,0] [0,0]
        if self.args.aggregation == 'first':
            lcor[0] = myids_cor
            ### exx   ample  #n_subtok=2
            ### [15,0] [0,0] #when n_subtokens=1 (aggregation is first)
        else:
            lcor[-1] = myids_cor
            ### exx   ample  #n_subtok=2
            ### [0,0] [15,0] #when n_subtokens=1 (aggregation is last)
        return lcor
    
    def __len__(self):
        return len(self.Data)

    def __iter__(self):
        assert len(self.Data) > 0, 'Empty dataset'
        logging.info('Shuffling dataset to build shards')
        idx_Data = [i for i in range(len(self.Data))]
        np.random.shuffle(idx_Data)
        self.args.shard_size = self.args.shard_size or len(idx_Data)
        shards = [idx_Data[i:i+self.args.shard_size] for i in range(0, len(idx_Data), self.args.shard_size)] # split dataset in shards
        logging.info('Built {} shards with up to {} examples'.format(len(shards),self.args.shard_size))
        for s,shard in enumerate(shards):
            logging.info('Building batchs for shard {}/{}'.format(s+1,len(shards)))
            batchs = self.build_batchs(shard)
            logging.info('Found {} batchs'.format(len(batchs)))
            for batch in batchs:
                yield self.format_batch(batch)
            logging.info('End of shard {}/{}'.format(s+1,len(shards)))
        logging.info('End of dataset')
            
    def build_batchs(self, shard):
        shard_len = [len(self.Data[idx]['ids_src']) for idx in shard]
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
            return len(self.Data[idx]['ids_src']) ### number of subwords
        return 1 ### number of sentences

        
    def format_batch(self, idxs):
        batch_raw = []
        batch_str_src = []
        batch_ids_src = []
        batch_ids_agg = []
        batch_ids_tag = []
        batch_ids_cor = []
        maxl = 0
        for idx in idxs:
            if 'str_src' not in self.Data[idx]:
                logging.warning('filtered {} line'.format(idx))
                continue
            batch_raw.append(self.Data[idx]['raw'])
            if maxl < len(self.Data[idx]['ids_src']):
                maxl = len(self.Data[idx]['ids_src'])
            batch_str_src.append(self.Data[idx]['str_src'])
            batch_ids_src.append(self.Data[idx]['ids_src'])
            batch_ids_agg.append(self.Data[idx]['ids_agg'])
            batch_ids_tag.append(self.Data[idx]['ids_tag'])
            batch_ids_cor.append(self.Data[idx]['ids_cor'])
        ### convert to tensor
        batch_ids_src = pad_listoflists(batch_ids_src,pad=self.idx_PAD_src,maxl=maxl)
        batch_ids_agg = pad_listoflists(batch_ids_agg,pad=-1,maxl=maxl)
        #if ids_tag/ids_cor are smaller than ids_src/ids_agg i add PAD so as to obtain the same size
        batch_ids_tag = pad_listoflists(batch_ids_tag,pad=self.idx_PAD_tag,maxl=maxl)
        batch_ids_cor = pad_listoflists(batch_ids_cor,pad=[self.idx_PAD_cor]*self.args.n_subtokens,maxl=maxl,n_subtokens=self.args.n_subtokens)
        if self.args.log == 'debug':
            debug_batch(idxs, batch_raw, batch_ids_src, batch_ids_agg, batch_ids_tag, batch_ids_cor)
        return [batch_ids_src, batch_ids_agg, batch_ids_tag, batch_ids_cor, idxs, batch_str_src]

    
    def reformat_batch(self, batch):
        batch_words = []
        batch_ids = []
        batch_ids2words = []
        maxl_ids = 0
        for i in range(len(batch)):
            ids = self.token.ids(batch[i], add_special_tokens=True, is_split_into_words=False)
            words, ids2words, _, _ = self.token.words_ids2words_subwords_lids(ids)
            #logging.debug('reformat {}'.format(batch[i]))
            #logging.debug('words {}'.format(words))
            #logging.debug('ids {}'.format(ids))
            #logging.debug('ids2words {}'.format(ids2words))
            batch_words.append(words)
            batch_ids.append(ids)
            batch_ids2words.append(ids2words)
            maxl_ids = len(ids) if len(ids) > maxl_ids else maxl_ids
        batch_ids = pad_listoflists(batch_ids, pad=self.idx_PAD_src, maxl=maxl_ids)
        batch_ids2words = pad_listoflists(batch_ids2words, pad=-1, maxl=maxl_ids)
        return [batch_ids, batch_ids2words, batch_words]
