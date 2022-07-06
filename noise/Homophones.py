import sys
import json
import time
import random
import pickle
import logging
import argparse
from collections import defaultdict
from utils.Utils import create_logger

class Homophones():
    
    def __init__(self,f):
       with open(f, "rb") as fdi:
           self.txt2txts = pickle.load(fdi)
           logging.info('Read {} with {} entries'.format(f,len(self.txt2txts)))

    def __call__(self, txt):
        txts = self.txt2txts[txt] if txt in self.txt2txts else None
        if txts is None or len(txts) == 0: 
            return None
        i = random.randint(0,len(txts)-1)
        return txts[i]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('hphones', type=str, default=None, help='Homophones (pickle) file')
    parser.add_argument('--log', type=str, default="info", help='Log level (info)')
    args = parser.parse_args()
    create_logger(None, args.log)
    logging.info("Options = {}".format(args.__dict__))
    
    create_logger(None,'info')
    hphones = Homophones(args.hphones)

    tic = time.time()
    n = 0
    for l in sys.stdin:
        n += 1
        ldict = json.loads(l) #list of words (dicts) with noise injected
        for d in ldict:
            if 't' in d: ### word is in lexicon
                txt = hphones(d)
                if txt is not None:
                    print("{}\t{}\t{}\t{}".format(d['t'],txt))
    toc = time.time()
    logging.info('Done {} lines ({:.2f} seconds)'.format(n,toc-tic))
