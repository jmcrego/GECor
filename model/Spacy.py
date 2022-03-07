#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import spacy
import logging
space = '~'

class Spacy():
    
    def __init__(self, model="fr_core_news_md", pos_to_consider=['NOUN','ADJ','VERB','AUX']):
        self.nlp = spacy.load(model)
        self.pos_to_consider = pos_to_consider
        logging.info('Loaded Spacy model={}'.format(model))

    def __call__(self, toks):
        tok2inf = {}
        tokens = self.nlp(' '.join(toks))
        for token in tokens:
            tok = token.text.replace(' ',space)
            if toks.count(tok) != 1:
                continue
            lem = token.lemma_
            if len(lem) == 0 or ' ' in lem:
                continue
            pos = str(token.pos_)
            if pos not in self.pos_to_consider:  ### only ADJ, NOUN and VERBs are considered (spacy tags)
                continue
            lem_pos_inf = lem+';'+pos

            INF = str(token.morph).replace('|',';').split(';')
            if 'NumType=Card' in INF:
                INF.remove('NumType=Card')
            if 'NumType=Ord' in INF:
                INF.remove('NumType=Ord')
            if '' in INF:
                INF.remove('')
                
            if len(INF):
                lem_pos_inf += ';' + ';'.join(INF)
            tok2inf[tok] = lem_pos_inf
        return tok2inf

if __name__ == '__main__':

    spacy = Spacy()

    for l in sys.stdin:
        print(spacy(l.rstrip().split()))
