#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from tqdm import tqdm
import logging
import argparse
from collections import defaultdict
from model.Tokenizer import Tokenizer
from model.Utils import create_logger
from model.Noiser import separ

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tags', action='store_true', help='Build vocabulary of tags from noisy examples')
    args = parser.parse_args()
    
    create_logger(None,'info')
    t = Tokenizer("flaubert/flaubert_base_cased")
    data = sys.stdin.readlines()
    logging.info('Read {} lines from stdin'.format(len(data)))

    word2freq = defaultdict(int)
    n_words = 0
    if args.tags:
        for l in tqdm(data):
            for w_t in l.rstrip().split('\t')[0].split():
                word, tag = w_t.split(separ)
                if '_' in tag:
                    tag, _ = tag.split('_')
                word2freq[tag] += 1
                n_words += 1            
    else:
        for l in tqdm(data):
            l = t.get_pretok(t.get_ids(l.rstrip()))
            for word in l.split():
                word2freq[word] += 1
                n_words += 1
                
    logging.info('Found {} tokens for a vocabulary of {} entries'.format(n_words,len(word2freq)))
    for word, freq in sorted(word2freq.items(), key=lambda kv:kv[1], reverse=True):
        print(word,freq)
    logging.info('Done')
