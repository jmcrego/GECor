#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from tqdm import tqdm
import logging
from collections import defaultdict
from model.Tokenizer import Tokenizer
from model.Utils import create_logger

if __name__ == '__main__':

    create_logger(None,'info')
    t = Tokenizer("flaubert/flaubert_base_cased")
    word2freq = defaultdict(int)
    
    data = sys.stdin.readlines()
    logging.info('Read {} lines from stdin'.format(len(data)))
    n_words = 0
    for l in tqdm(data):
        l = t.get_pretok(t.get_ids(l.rstrip()))
        for word in l.split():
            word2freq[word] += 1
            n_words += 1            
    logging.info('Found {} words'.format(n_words))
    for word, freq in sorted(word2freq.items(), key=lambda kv:kv[1], reverse=True):
        print(word,freq)
    logging.info('Done')
