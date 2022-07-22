#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import gzip
import json
import copy
import random
import string
import logging

KEEP = '$KEEP'
PAD = '<pad>' #### pad token for error/corr layers
SEPAR1 = '￤'
SEPAR2 = '～'
MAX_IDS_LEN = 500
PUNCTUATION = [c for c in "!?,.:;\"'()[]{}@<=>*%+-^~|&#\/"]
CURRENCY = [c for c in "$£€¥₩₽Ξ¢"]

def create_logger(logfile, loglevel):
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        logging.error("Invalid log level={}".format(loglevel))
        sys.exit()
    if logfile is None or logfile == 'stderr':
        logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=numeric_level)
        logging.debug('Created Logger level={}'.format(loglevel))
    else:
        logging.basicConfig(filename=logfile, format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=numeric_level)
        logging.debug('Created Logger level={} file={}'.format(loglevel, logfile))

def shape_of_word(txt):
    shape = []
    for c in txt:
        if c.isalpha():
            if c.isupper():
                case = 'X'
            elif c.islower():
                case = 'x'
            else:
                case = 'z' # character is alpha but neither upper nor lower. Ex: د ه ش 扶 桑
        elif c.isdecimal():
            case = 'd'
        elif c in PUNCTUATION or c in CURRENCY:
            case = c
        else:
            case = 'o'
        if len(shape) == 0 or shape[-1] != case:
            shape.append(case)
    return ''.join(shape)

def reshape(txt, shape):
    if shape == 'Xx':
        if len(txt) > 1:
            txt = txt[0].upper() + txt[1:]
        elif len(txt) == 1:
            txt = txt[0].upper()
        else:
            pass
    elif shape == 'x':
        txt = txt.lower()
    elif shape == 'X':
        txt = txt.upper()
    else:
        pass
    return txt

def get_linear(lwdict, lexicon, p_cor=0.0, vocab=None):
    toks = []
    for wdict in lwdict:
        tok = []
        tok.append(wdict['W'])
        tok.append(SEPAR2.join(list(map(str,wdict['iW']))))
        tok.append(shape_of_word(tok[0]))
        tok.append(str(tok[0] in lexicon))
        tok.append(wdict['E'] if 'E' in wdict else KEEP)

        C = None
        if 'C' in wdict:
            C = wdict['C']
            iC = wdict['iC']
        else:
            if random.random() < p_cor:
                C = wdict['W']
                iC = vocab[C] if vocab is not None else wdict['iW']

        if C is not None:
            tok.append(C)
            if isinstance(iC,list):
                tok.append(SEPAR2.join(list(map(str,iC))))
            else:
                tok.append(str(iC))

        toks.append(SEPAR1.join(tok))
    return ' '.join(toks)

def build_sentence(toks, lids, nlp=None):
    s = [{'W': toks[idx], 'iW': lids[idx]} for idx in range(len(toks))]
    if nlp is not None:
        tokens = nlp(' '.join(toks))
        assert(len(tokens) == len(toks))
        for idx,token in enumerate(tokens):
            assert(token.text == toks[idx])
            ling = {'pos':str(token.pos_), 'lem':token.lemma_}
            for inf in str(token.morph).split('|'):
                if '=' in inf:
                    key, val = inf.split('=')
                    ling[key] = val
            s[idx]['l'] = ling
    return s

def Word(wrd, wrd_ids, err=None, cor=None, cor_ids=None):
    d = {'W': copy.deepcopy(wrd), 'iW':copy.deepcopy(wrd_ids)}
    if err is not None:
        d['E'] = copy.deepcopy(err)
    if cor is not None:
        d['C'] = copy.deepcopy(cor)
    if cor_ids is not None:
        d['iC'] = copy.deepcopy(cor_ids)
    return d

def conll(l):
    out = []
    for w in l:
        wout = []
        if 'raw' in w:
            wout.append(w['raw'])
        if 'iraw' in w:
            wout.append("iraw:{}".format('-'.join(list(map(str,w['iraw'])))))
        if 'shp' in w:
            wout.append('shp:'+w['shp'])
        if 'ishp' in w:
            wout.append('ishp:'+str(w['ishp']))
        if 'lex' in w:
            wout.append('lex:'+str(w['lex']!=''))
        if 'err' in w:
            wout.append('err:'+w['err'])
        if 'ierr' in w:
            wout.append('ierr:'+str(w['ierr']))
        if 'cor' in w:
            wout.append('cor:'+w['cor'])
        if 'icor' in w:
            wout.append('icor:'+str(w['icor']))
        if 'iCOR' in w:
            wout.append("iCOR:{}".format('-'.join(list(map(str,w['iCOR'])))))
        if 'lng' in w:
            wout.append('lng:'+w['lng'])
        if 'ilng' in w:
            wout.append('ilng:'+str(w['ilng']))
#        if 'plm' in w:
#            wout.append(w['plm'])
        out.append('\t'.join(wout))
    return '\n'.join(out) + '\n'
