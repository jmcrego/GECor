import sys
import time
import json
import logging
import argparse
from utils.Spacyfy import Spacyfy
from utils.Utils import create_logger

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, nargs='*', help='Input files [do not use for STDIN]')
    parser.add_argument('--lex',   type=str, default=None, help='Lexicon (pickle) file [required]', required=True)
    parser.add_argument('--voc_shapes', type=str, default=None, help='Vocabulary (shapes) file [required]', required=True)
    parser.add_argument('-o',   type=str, default=None, help='Output file [do not use for STDOUT]')
    parser.add_argument('-m',   type=str, default="fr_core_news_md", help='Spacy model name (fr_core_news_md)')
    parser.add_argument('-b',   type=int, default=1000, help='Batch size (1000)')
    parser.add_argument('-n',   type=int, default=16, help='Parallel processes unless only_tokenize (16)')
    parser.add_argument('-only_tokenize', action='store_true', help='Only tokenize')
    parser.add_argument('-log', type=str, default="info", help='Logging level [critical, error, warning, info, debug] (info)')
    args = parser.parse_args()
    if args.only_tokenize:
        args.n = 1
    if len(args.file) == 0:
        args.file.append('stdin')
    logfile = 'stderr' if args.o is None else args.o+".log"
    create_logger(logfile,args.log)
    logging.info("Options = {}".format(args.__dict__))
    
    spacyfy = Spacyfy(args.m, args.b, args.n, args.only_tokenize, args.lex, args.voc_shapes)    
    nlines = 0
    proc_time = 0.00001
    with open(args.o, "w") if args.o is not None else sys.stdout as fdo:
        for fn in args.file:
            spacyfy.read_lines(fn)
            tic = time.time()
            for line in spacyfy:
                fdo.write(json.dumps(line, ensure_ascii=False) + '\n')
                nlines += 1
            toc = time.time()
            proc_time += toc - tic
    logging.info('Done ({:.2f} seconds, {:.2f} lines/second)'.format(proc_time, nlines/proc_time))


    
