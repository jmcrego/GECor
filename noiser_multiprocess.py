#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import copy
import random
import unicodedata
import time
#from psutil import cpu_count
from model.Keyboard import Keyboard
from model.Lexicon import Lexicon
from model.Vocab import Vocab
from model.Spacy import Spacy
from model.Tokenizer import Tokenizer
from collections import defaultdict
from transformers import FlaubertTokenizer
import logging
from model.Utils import create_logger
import multiprocessing as mp
from model.Noiser import Noiser
import argparse

separ = '￨'
keep = '·'
used = '׃'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--json', help='Noiser config file (required)', required=True)
    parser.add_argument('--input', help='input file', required=True)
    args = parser.parse_args()

    create_logger(None,'info')
        
    ### Multi_process_setup
    cpu_count_ = mp.cpu_count() # number of available cpus
    sys.stderr.write("%d cpus are available"%cpu_count_)
    file_name = args.input
    noiser_config = args.json
    file_size = os.path.getsize(file_name)
    chunk_size = file_size // cpu_count_
    chunk_args = []
    start_time = time.time()
    count = 0
    ### Apply noise to a line
    def process_line(noiser, line):
        #line = line.encode('utf8').decode('utf8')
        l = l.rstrip()
        ids = noiser.tokenizer.get_ids(l)
        words, ids2words, _ = noiser.tokenizer.get_words_ids2words_subwords(ids)
        noisy_words, noisy_tags, noisy_idx2lids = noiser(l, words, ids, ids2words, n+1)
        noisy_wordstags = [noisy_words[i]+separ+(noisy_tags[i].replace(used,keep)) for i in range(len(noisy_words))]
        print('{}\t{}'.format(' '.join(noisy_wordstags), noisy_idx2lids))

        l = line.rstrip()
        ids = noiser.tokenizer.get_ids(l)
        words, ids2words, _ = noiser.tokenizer.get_words_ids2words_subwords(ids)
        noisy_words, noisy_tags, noisy_idx2lids = noiser(l, words, ids, ids2words, 1)
        noisy_wordstags = [noisy_words[i]+separ+(noisy_tags[i].replace(used,keep)) for i in range(len(noisy_words))]
        print(('{}\t{}'.format(' '.join(noisy_wordstags), noisy_idx2lids)), flush=True)
                
    ### Apply noise to a chunk of the original file and return the statistic of noise used in that chunk
    def process_chunk(file_name, noiser_config, chunk_start, chunk_end):
        noiser = Noiser(noiser_config)
        
        with open(file_name, 'r') as f:
            # Moving stream position to `chunk_start`
            f.seek(chunk_start)
            count = 0
            # Read and process lines until `chunk_end`
            for line in f:
                chunk_start += len(line)
                if chunk_start > chunk_end:
                    break
                process_line(noiser, line)
                count +=1
        return noiser.stats_noise_injected, noiser.stats_n_injected, count

    ### Split original file to many chunks of chunk_size
    with open(file_name, 'r', encoding="latin1") as f:
        def is_start_of_line(position):
            if position==0:
                return True
            f.seek(position-1)
            return f.read(1)=='\n'
        
        def get_next_line_position(position):
            f.seek(position)
            f.readline()
            return f.tell()
        
        chunk_start = 0
        sent_id = 0
        while chunk_start<file_size:
            chunk_end = min(file_size,chunk_start+chunk_size)

            while not is_start_of_line(chunk_end):
                chunk_end -=1
            
            # Handle the case when a line is too long to fit the chunk size
            if chunk_start == chunk_end:
                chunk_end = get_next_line_position()
    
            # Save `process_chunk` arguments
            args = (file_name, noiser_config, chunk_start, chunk_end)
            chunk_args.append(args)

            chunk_start = chunk_end
    
    ### Set up and launch noiser process on each cpu
    with mp.Pool(cpu_count_) as p:
        # Run chunks in parallel
        chunk_results = p.starmap(process_chunk, chunk_args)

    # Combine chunks' statistics into `results`
    stats_noise_injected = {}
    stats_n_injected = {}
    
    for chunk_result in chunk_results:
      chunk_stats_noise_injected, chunk_stats_n_injected, count_ = chunk_result
      for item in chunk_stats_noise_injected.items():
        stats_noise_injected[item[0]] = stats_noise_injected.get(item[0],0) + item[1]
      for item in chunk_stats_n_injected.items():
        stats_n_injected[item[0]] = stats_n_injected.get(item[0],0) + item[1]
      count += count_
    
    sys.stderr.write('N\tNoise_type\n')
    for noise, n in sorted(stats_noise_injected.items(), key=lambda x: x[1], reverse=True):
        sys.stderr.write("{}\t{}\n".format(n,noise))
    sys.stderr.write('N_noise\tN\n')
    for k, n in sorted(stats_n_injected.items()):
        sys.stderr.write("{}\t{}\n".format(k,n))
    sys.stderr.write("finish after %f s \n"%(time.time()-start_time))
    sys.stderr.write("%f sentence per second"%(count/(time.time()-start_time)))