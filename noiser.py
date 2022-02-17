#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
import logging
import argparse
from model.Noiser import Noiser, separ, keep, used
from model.Utils import create_logger

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--json', help='Noiser config file (required)', required=True)
    args = parser.parse_args()

    create_logger(None,'info')
    noiser = Noiser(args.json)

    for n,l in enumerate(sys.stdin):
        l = l.rstrip()
        ids = noiser.tokenizer.get_ids(l)
        words, ids2words, _ = noiser.tokenizer.get_words_ids2words_subwords(ids)
        noisy_words, noisy_tags, noisy_idx2lids = noiser(l, words, ids, ids2words, n+1)
        noisy_wordstags = [noisy_words[i]+separ+(noisy_tags[i].replace(used,keep)) for i in range(len(noisy_words))]
        print('{}\t{}'.format(' '.join(noisy_wordstags), noisy_idx2lids))
    noiser.stats()
        

