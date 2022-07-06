import sys
import time
import json
from tqdm import tqdm
import logging
import argparse
from collections import defaultdict
from utils.Spacyfy import Spacyfy
from utils.Utils import create_logger

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, nargs='*', help='Input files [do not use for STDIN]')
    parser.add_argument('--lex',type=str, default=None, help='Lexicon (pickle) file [required]', required=True)
    parser.add_argument('-o',   type=str, default=None, help='Output file [required]', required=True)
    parser.add_argument('-m',   type=str, default="fr_core_news_md", help='Spacy model name (fr_core_news_md)')
    parser.add_argument('-b',   type=int, default=1000, help='Batch size (1000)')
#    parser.add_argument('-n',   type=int, default=16, help='Parallel processes unless only_tokenize (16)')
#    parser.add_argument('-s',   type=str, default="‗", help='Replace spaces by this char (‗)')
    parser.add_argument('-log', type=str, default="info", help='Logging level [critical, error, warning, info, debug] (info)')
    args = parser.parse_args()
    if len(args.file) == 0:
        args.file.append('stdin')
    create_logger(None,args.log)
    logging.info("Options = {}".format(args.__dict__))
    words = defaultdict(int)
    shapes = defaultdict(int)
    spacyfy = Spacyfy(args.m, args.b, 1, True, args.lex) #True indicates only tokenize, 1 indicates single process
    nlines = 0
    proc_time = 0.00001
    for fn in args.file:
        spacyfy.read_lines(fn)
        tic = time.time()
        for line in tqdm(spacyfy):
            for d in line:
                if d['r'].isnumeric() or '‗' in d['r']:
                    continue
                words[d['r']] += 1
                shapes[d['s']] += 1
                #if 'plm' in d:
                #    l = d['plm'].split('|')
                #    l.pop(1) #discard lemma
                #    features['|'.join(l)] += 1
        nlines += 1
    toc = time.time()
    proc_time += toc - tic

    with open(args.o + '.CORRECTIONS.freq', "w") as fdo:
        for k,v in sorted(words.items(), key=lambda kv: kv[1], reverse=True):
            fdo.write('{}\t{}\n'.format(v,k))
            
    with open(args.o + '.SHAPES.freq', "w") as fdo:
        for k,v in sorted(shapes.items(), key=lambda kv: kv[1], reverse=True):
            fdo.write('{}\t{}\n'.format(v,k))
        
    with open(args.o + '.ERRORS', "w") as fdo:
        fdo.write("$APND\n$CASE1\n$CASEn\n$DELE\n$HYPHm\n$HYPHs\n$KEEP\n$LEMM\n$MRGE\n$PHON\n$SPEL\n$SPLT\n$SWAP\n")
        
    logging.info('Found {} words, {} shapes in ({:.2f} seconds, {:.2f} lines/second)'.format(len(words), len(shapes), proc_time, nlines/proc_time))


    
