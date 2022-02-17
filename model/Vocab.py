#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import logging
from collections import defaultdict

class Vocab():
    def __init__(self, f):
        self.PAD = '<PAD>'
        self.UNK = '<UNK>'
        self.vocab = defaultdict()
        self.VOCAB = []
        self.idx_PAD = 0
        self.vocab[self.PAD] = len(self.vocab)
        self.VOCAB.append(self.PAD)
        self.idx_UNK = 1
        self.vocab[self.UNK] = len(self.vocab)
        self.VOCAB.append(self.UNK)
        with open(f,'r') as fd:
            for l in fd:
                wrd = l.rstrip()
                if wrd in self.vocab:
                    logging.info('Repeated entry {} in vocab'.format(wrd))
                    continue
                self.vocab[wrd] = len(self.vocab)
                self.VOCAB.append(wrd)
        logging.info('Read Vocab={} with {} entries'.format(f,len(self.vocab)))

    def __contains__(self, s):
        return s in self.vocab
        
    def __getitem__(self, s):
        ### return a string
        if type(s) == int:
            return self.VOCAB[s]
        ### return an index
        if s in self.vocab:
            return self.vocab[s]
        return self.idx_UNK

    def __len__(self):
        return len(self.vocab)

    
if __name__ == '__main__':
                    
    l = Vocab('resources/french.dic.50k')
