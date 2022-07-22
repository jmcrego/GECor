import sys
import json
import time
import spacy
#from spacy.tokens import Doc
import random
import logging
import argparse
from collections import defaultdict
from noise.Noiser import Noiser
#from utils.FlaubertTok import FlaubertTok
#from utils.Lexicon import Lexicon
from utils.Utils import create_logger#, MAX_IDS_LEN, KEEP, get_linear, shape_of_word, reshape, build_sentence, Word, PUNCTUATION
#from noise.LemRules import LemRules
#from noise.Misspell import Misspell
#from model.Vocab import Vocab


if __name__ == '__main__': 

    parser = argparse.ArgumentParser()
    parser.add_argument('--lex', type=str, default=None, help='Lexicon (pickle) file [required]', required=True)
    parser.add_argument('--pho', type=str, default=None, help='Homophones (pickle) file [required]', required=True)
    parser.add_argument('--lem', type=str, default=None, help='LemRules config file [required]', required=True)
    #
    parser.add_argument('--errors', type=str, default=None, help='Error vocabulary (None) [required]', required=True)
    parser.add_argument('--correc',  type=str, default=None, help='Correction vocabulary (None) [required]', required=True)
    parser.add_argument('--lfeats', type=str, default=None, help='Linguistic features vocabulary (None) [required]', required=True)
    parser.add_argument('--shapes', type=str, default=None, help='Shape vocabulary (None) [required]', required=True)
    parser.add_argument('--inlexi', type=str, default=None, help='Inlexicon vocabulary (None) [required]', required=True)
    #
    parser.add_argument('--max_n', type=int, default=7, help='Maximum number of noises per sentence (7)')
    parser.add_argument('--max_r', type=float, default=0.3, help='Maximum ratio of noises/words per sentence (0.3)')
    parser.add_argument('--p_cor', type=float, default=0.5, help='probability of using cor layer without err (0.5)')
    parser.add_argument('--p_lng', type=float, default=0.5, help='probability of using lng layer without err (0.5)')
    #
    parser.add_argument('--w_phon', type=int, default=4, help='Weight of PHON noise [err/cor] (4)')
    parser.add_argument('--w_lemm', type=int, default=6, help='Weight of LEMM noise [err/cor] (6)')
    parser.add_argument('--w_spel', type=int, default=2, help='Weight of SPEL noise [err/cor] (2)')
    parser.add_argument('--w_apnd', type=int, default=1, help='Weight of APND noise [err/cor] (1)')
    parser.add_argument('--w_splt', type=int, default=1, help='Weight of SPLT noise [err/cor] (1)')
    parser.add_argument('--w_dele', type=int, default=1, help='Weight of DELE noise [err] (1)')
    parser.add_argument('--w_mrge', type=int, default=1, help='Weight of MRGE noise [err] (1)')
    parser.add_argument('--w_hyph', type=int, default=5, help='Weight of HYPH noise [err] (5)')
    parser.add_argument('--w_swap', type=int, default=1, help='Weight of SWAP noise [err] (1)')
    parser.add_argument('--w_case', type=int, default=1, help='Weight of CASE noise [err] (1)')
    #
    parser.add_argument('--seed', type=int, default=0, help='Seed for randomness (0)')
    parser.add_argument('--log', type=str, default="info", help='Log level (info)')
    args = parser.parse_args()
    create_logger(None, args.log)
    logging.info("Options = {}".format(args.__dict__))

    noises = []
    for k,v in args.__dict__.items():
        if k.startswith('w_'):
            noises.append([k[2:], v])
    
    noiser = Noiser(noises,args.lem,args.lex,args.pho,args.errors,args.correc,args.lfeats,args.shapes,args.inlexi,args.max_n,args.max_r,args.p_cor,args.p_lng,args.seed)

    n_sents = 0
    tic = time.time()
    for l in sys.stdin:
        ldict = noiser(json.loads(l))
        print(json.dumps(ldict, ensure_ascii=False))
        n_sents += 1
        if n_sents%10000 == 0:
            noiser.debug(tic)
    logging.info('Done')
    noiser.debug(tic)
