#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import logging
from collections import defaultdict

class Lexicon():
    def __init__(self, f, separ='￨'):
        self.separ = separ
        self.lem2wrd = defaultdict(set)
        self.wrd2lem = defaultdict(set)
        self.pho2wrd = defaultdict(set)
        self.wrd2pho = defaultdict(set)
        self.wrd2pos = defaultdict(set)
        self.lempos2wrds = defaultdict(set)
        self.pos2wrds = defaultdict(set) #only for prep and art
        self.wrd2wrds_same_lemma = defaultdict(set)
        self.wrd2wrds_same_pos = defaultdict(set) #only for prep and art
        self.wrd2wrds_homophones = defaultdict(set)
        self.pos = set()
        if 'Lexique383' in f:
            self.read_Lexique383(f)
        elif 'Morphalou3.1' in f:
            self.read_Morphalou31_CSV(f)
        else:
            logging.error('error: unparsed file {}'.format(f))
            sys.exit()
        self.build_wrd2wrds()
            
    def read_Lexique383(self, f):
        with open(f,'r') as fd:
            for l in fd:
                toks = l.rstrip().split('\t')
                if len(toks) < 3:
                    continue
                wrd, pho, lem, pos = toks[0], toks[1], toks[2], toks[3]
                if ' ' in wrd or ' ' in lem or ' ' in pho or ' ' in pos:
                    continue
                self.add(wrd,lem,pos,pho)
        logging.info('Read Lexicon={} with {} words {} lemmas {} phones'.format(f,len(self.wrd2lem),len(self.lem2wrd),len(self.pho2wrd)))
        
    def read_Morphalou31_CSV(self, f):
        with open(f,'r') as fd:
            for l in fd:
                toks = l.rstrip().split(';')
                if len(toks) <= 16:
                    #sys.stderr.write('Discarded entry: {}\n'.format(toks))
                    continue
                if len(toks[0]):
                    lem = toks[0]
                    pos = toks[2]
                    secondpos = toks[3]
                    wrd = toks[0] ### same as lemma
                    pho = toks[7]
                else:
                    wrd = toks[9]
                    pho = toks[16]
                pos = pos.replace(' ','')
                if ' ' in lem:
                    #sys.stderr.write('Discarded lem: {}\t{}\n'.format(lem,toks))
                    continue
                if ' ' in wrd:
                    #sys.stderr.write('Discarded wrd: {}\t{}\n'.format(wrd,toks))
                    continue
                if len(pos) == 0:
                    #sys.stderr.write('Discarded pos: {}\t{}\n'.format(pos,toks))
                    continue
                    
                for pho in  pho.replace(' ','').split('OU'):
                    self.add(wrd,lem,pos,pho,secondpos)
        logging.info('Read Lexicon:{} with {} words {} lemmas {} phones'.format(f,len(self.wrd2lem),len(self.lem2wrd),len(self.pho2wrd)))

                
    def add(self, wrd, lem, pos, pho, secondpos=''):
        #use same tags as SpaCy for verbs, nouns adjectives and adverbs
        if pos == 'VER' or pos == 'Verbe':
            pos = 'VERB' 
        if pos == 'Nomcommun' or pos == 'NOM':
            pos = 'NOUN' 
        if pos == 'Adjectifqualificatif':
            pos = 'ADJ'
        if pos == 'Adverbe':
            pos = 'ADV'
        if pos.startswith('PRO:rel'):
            pos = 'PROREL'
        if pos.startswith('ART'):
            pos = 'ART'
        if pos == 'Pronom' and secondpos == 'relatif':
            pos = 'PROREL'
            print(wrd,pos)

        if wrd == 'la-la-la':
            return
        
        self.lem2wrd[lem].add(wrd)
        self.wrd2lem[wrd].add(lem)
        self.pos.add(pos)
        self.wrd2pos[wrd].add(pos)
        self.lempos2wrds[lem+self.separ+pos].add(wrd)
        if pos in ['PROREL', 'ART']:
            self.pos2wrds[pos].add(wrd)
        if len(pho):
            self.pho2wrd[pho].add(wrd)
            self.wrd2pho[wrd].add(pho)
        
    def build_wrd2wrds(self):
        for wrd in self.wrd2lem:
            for lem in self.wrd2lem[wrd]:
                for w in self.lem2wrd[lem]:
                    if w != wrd:
                        self.wrd2wrds_same_lemma[wrd].add(w)

        for wrd in self.wrd2pho:
            for pho in self.wrd2pho[wrd]:
                for w in self.pho2wrd[pho]:
                    if w != wrd:
                        self.wrd2wrds_homophones[wrd].add(w)

        for pos in self.pos2wrds:
            wrds = self.pos2wrds[pos]
            for wrd in wrds:
                for wrd2 in wrds:
                    if wrd == wrd2:
                        continue
                    self.wrd2wrds_same_pos[wrd].add(wrd2)

        
if __name__ == '__main__':

                    
    l = Lexicon('resources/Morphalou3.1_CSV.csv') #'resources/Lexique383.tsv'
    for p in l.pos:
        print(p)
    print('#words: {}'.format(len(l.wrd2lem)))
    print('#lemmas: {}'.format(len(l.lem2wrd)))
    print('#phones: {}'.format(len(l.pho2wrd)))