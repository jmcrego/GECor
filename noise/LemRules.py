import sys
import json
import time
import random
import logging
import argparse
from collections import defaultdict
from utils.Lexicon import Lexicon
from utils.Utils import create_logger

class LemRules():
    
    def __init__(self,frules,lexicon):
        self.lex = lexicon
        self.rules = defaultdict(list)
        with open(frules,"r") as f:
            for l in f: #lemendswith=er,pos=VER,tense=infinitive;lemendswith=er,pos=VER,mode=participle,tense=past
                l = l.rstrip()
                if l.startswith('#') or len(l) == 0:
                    continue
                toks = l.split(';')
                #print(toks)
                for i in range(len(toks)):
                    left = toks[i]  #"lemendswith=er,pos=VER,tense=infinitive"
                    right = toks[:] #["lemendswith=er,pos=VER,tense=infinitive", "lemendswith=er,pos=VER,mode=participle,tense=past"]
                    #right.pop(i)
                    self.rules[left] = right
        logging.info('Read {} with {} rules'.format(frules,len(self.rules)))

    def match(self, siderule, pos, lem, mode, tense, pers, nombre, genre):
        #print("siderule={}".format(siderule))
        #print("match pos={} lem={} mode={} tense={} pers={} nombre={} genre={} siderule={}".format(pos, lem, mode, tense, pers, nombre, genre, siderule))
        for condition in siderule.split(','): #lemendswith=er,pos=VER,tense=infinitive
            #print("condition={}")
            csrc,ctgt = condition.split('=')
            if csrc == 'lemendswith':
                if not lem.endswith(ctgt):
                    return False
            elif csrc == 'pos':
                if not ctgt == pos:
                    return False
            elif csrc == 'tense':
                if not ctgt == tense:
                    return False
            elif csrc == 'mode':
                if not ctgt == mode:
                    return False
            elif csrc == 'pers':
                if not ctgt == pers:
                    return False
            elif csrc == 'nombre':
                if not ctgt == nombre:
                    return False
            elif csrc == 'genre':
                if not ctgt == genre:
                    return False
        return True

    def features(self, plm):
        toks = plm.split('|')
        if len(toks) != 7:
            logging.error('bad number of features in plm={}'.format(plm))
            sys.exit()
        return toks

    def __call__(self, d):
        itxt = d['t']
        iplm = d['plm']
        ipos, ilem, imode, itense, ipers, inombre, igenre = self.features(iplm)
        ltxt, lplm = [], [] #found words and their features
        for leftrule, rightrules in self.rules.items():
            if self.match(leftrule, ipos, ilem, imode, itense, ipers, inombre, igenre):
                #logging.debug('\imatch\t{}\t{}'.format(iplm,leftrule))
                for rightrule in rightrules:
                    for rtxt in self.lex.lempos2txt[ilem+ipos]: ### replacement rtxt must have the same lem AND pos than itxt
                        if rtxt != itxt: ### cannot be equal
                            for rplm in self.lex.txtlempos2feats[rtxt+ilem+ipos]:
                                rpos, rlem, rmode, rtense, rpers, rnombre, rgenre = self.features(rplm)
                                if rlem == ilem and rpos == ipos: ### must be same lem AND pos
                                    if self.match(rightrule, rpos, rlem, rmode, rtense, rpers, rnombre, rgenre):
                                        #logging.debug('\trmatch\t{}\t{}'.format(rplm,rightrule))
                                        ltxt.append(rtxt)
                                        lplm.append(rplm)
        if len(ltxt) == 0:
            return None, None
        i = random.randint(0,len(ltxt)-1)
        return ltxt[i], lplm[i]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('lex', type=str, default=None, help='Lexicon (pickle) file')
    parser.add_argument('rules', type=str, default=None, help='Rules file')
    parser.add_argument('--log', type=str, default="info", help='Log level (info)')
    args = parser.parse_args()
    create_logger(None, args.log)
    logging.info("Options = {}".format(args.__dict__))
    
    create_logger(None,'info')
    lex = Lexicon(args.lex)
    rules = LemRules(args.rules,lex)

    tic = time.time()
    n = 0
    for l in sys.stdin:
        n += 1
        ldict = json.loads(l) #list of words (dicts) with noise injected
        for d in ldict:
            if 't' in d and 'plm' in d: ### word is in lexicon and has pos,lem,morph
                txt, plm = rules(d)
                if txt is not None:
                    iplm = d['plm'].split('|')
                    rplm = plm.split('|')
                    del iplm[1] #del lem
                    del rplm[1] #del lem
                    print("{}\t{}\t{}\t{}".format(d['t'],txt,'|'.join(iplm),'|'.join(rplm)))
    toc = time.time()
    logging.info('Done {} lines ({:.2f} seconds)'.format(n,toc-tic))
