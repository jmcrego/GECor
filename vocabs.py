import sys
import time
import json
import logging
import argparse
from tqdm import tqdm
from collections import defaultdict
from utils.Spacyfy import Spacyfy
from utils.Utils import create_logger

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, nargs='*', help='Input raw files [if not used reads STDIN]')
    parser.add_argument('--lex',type=str, default=None, help='Lexicon (pickle) file [required]', required=True)
    parser.add_argument('--one_out_of',type=int, default=100, help='Consider one sentence out of this many (100)')
    parser.add_argument('-o',   type=str, default=None, help='Output file [required]', required=True)
    parser.add_argument('-m',   type=str, default="fr_core_news_md", help='Spacy model name (fr_core_news_md)')
    parser.add_argument('-b',   type=int, default=1000, help='Batch size (1000)')
    parser.add_argument('-log', type=str, default="info", help='Logging level [critical, error, warning, info, debug] (info)')
    args = parser.parse_args()
    if len(args.file) == 0:
        args.file.append('stdin')
    create_logger(None,args.log)
    logging.info("Options = {}".format(args.__dict__))

    words = defaultdict(int)
    shapes = defaultdict(int)
    inlexi = defaultdict(bool)
    spacyfy = Spacyfy(args.m, args.b, 1, True, args.lex) #True indicates only tokenize, 1 indicates single process

    nlines = 0
    proc_time = 0.00001
    for fn in args.file:
        spacyfy.read_lines(fn, one_out_of=args.one_out_of)
        tic = time.time()
        for line in tqdm(spacyfy):
            nlines += 1
            for d in line:
                if d['raw'].isnumeric() or '‗' in d['raw']:
                    continue
                words[d['raw']] += 1
                shapes[d['shp']] += 1
                inlexi[('lex' in d and d['lex'] is not None)] += 1
    toc = time.time()
    proc_time += toc - tic

    with open(args.o + '.CORREC.freq', "w") as fdo:
        for k,v in sorted(words.items(), key=lambda kv: kv[1], reverse=True):
            fdo.write('{}\t{}\n'.format(v,k))
            
    with open(args.o + '.SHAPES.freq', "w") as fdo:
        for k,v in sorted(shapes.items(), key=lambda kv: kv[1], reverse=True):
            fdo.write('{}\t{}\n'.format(v,k))

    with open(args.o + '.INLEXI.freq', "w") as fdo:
        for k,v in sorted(inlexi.items(), key=lambda kv: kv[1], reverse=True):
            fdo.write('{}\t{}\n'.format(v,k))
            
    with open(args.o + '.ERRORS', "w") as fdo:
        fdo.write("$APND\n$CAS1\n$CASn\n$DELE\n$HYPm\n$HYPs\n$KEEP\n$LEMM\n$MRGE\n$PHON\n$SPEL\n$SPLT\n$SWAP\n")
        
    logging.info('Found {} words, {} shapes in ({:.2f} seconds, {:.2f} lines/second)'.format(len(words), len(shapes), proc_time, nlines/proc_time))


    
