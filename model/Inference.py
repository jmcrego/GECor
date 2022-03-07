# -*- coding: utf-8 -*-

import sys
import os
import logging
import numpy as np
import torch
import time
from model.GECor import save_checkpoint
from model.Noiser import separ, keep, used
from collections import defaultdict

class Inference():

    def __init__(self, model, testset, words, tags, lex, voc, idx_PAD, args, device):
        super(Inference, self).__init__()
        self.tags = tags
        self.words = words
        self.lex = lex
        self.voc = voc
        self.args = args
        self.corrected_sentences = [None] * len(testset)
        model.eval()
        with torch.no_grad():
            dinputs = {}
            for inputs, indexs, words, idxs in testset:
                inputs = inputs.to(device)
                indexs = indexs.to(device)
                dinputs['input_ids'] = inputs
                outtag, outwrd = model(dinputs, indexs) ### [bs, l, ts], [bs, l, ws]
                kbest_tags = outtag.argsort(dim=2, descending=True)[:,:,:self.args.Kt] #[bs, l, Kt]
                kbest_wrds = outwrd.argsort(dim=2, descending=True)[:,:,:self.args.Kw] #[bs, l, Kw]
                for i in range(kbest_tags.shape[0]):
                    self.sentence(idxs[i],words[i],kbest_tags[i],kbest_wrds[i])

        for i in range(len(self.corrected_sentences)):
            print(' '.join(self.corrected_sentences[i]))

            
    def sentence(self,idx,words,kbest_tags,kbest_wrds):
        #kbest_tags [l,Kt]
        #kbest_wrds [l,Kw]
        logging.debug('idx: {}'.format(idx))
        corrected_words = []
        for i in range(1,len(words)-1):
            if words[i] == "": ### has been deleted
                continue
            curr_tags = [self.tags[tag_idx] for tag_idx in kbest_tags[i].tolist()]
            curr_wrds = [self.words[wrd_idx] for wrd_idx in kbest_wrds[i].tolist()]
            logging.debug("{}\t{}\t{}".format(words[i], curr_tags, curr_wrds))
            corrected_word, delete_next = self.correct(words[i], curr_wrds,curr_tags, word_next=words[i+1] if i+1<len(words) else None)
            corrected_words.append(corrected_word)
            if delete_next:
                words[i+1] = ""
        self.corrected_sentences[idx] = corrected_words


    def find_inflection(self, word, curr_wrds):
        logging.debug('\tfind_inflection({})'.format(word))
        if word in self.lex.wrd2wrds_same_lemma:
            words_same_lemma = self.lex.wrd2wrds_same_lemma[word]
            logging.debug('\tinflections: {}'.format(words_same_lemma))
            for i, wrd in enumerate(curr_wrds):
                if wrd in words_same_lemma and wrd != word:
                    logging.debug('\tfirst inflection: {}'.format(curr_wrds[i]))
                    return i
        return -1

    
    def find_samepos(self, word, curr_wrds):
        logging.debug('\tfind_samepos({})'.format(word))
        if word in self.lex.wrd2wrds_same_pos:
            words_same_pos = self.lex.wrd2wrds_same_pos[word]
            logging.debug('\tsamepos: {}'.format(words_same_pos))
            for i, wrd in enumerate(curr_wrds):
                if wrd in words_same_pos and wrd != word:
                    logging.debug('\tfirst samepos: {}'.format(curr_wrds[i]))
                    return i
        return -1

    
    def find_homophone(self, word, curr_wrds):
        logging.debug('\tfind_homophone({})'.format(word))
        if word in self.lex.wrd2wrds_homophones:
            words_homophones = self.lex.wrd2wrds_homophones[word]
            logging.debug('\thomophones: {}'.format(words_homophones))
            for i, wrd in enumerate(curr_wrds):
                if wrd in words_homophones and wrd != word:
                    logging.debug('\tfirst homophone: {}'.format(curr_wrds[i]))
                    return i
        return -1


    def is_spell(self, wrd1, wrd2):
        swrd1 = set(wrd1.split())
        swrd2 = set(wrd2.split())
        if len(swrd1-swrd2) <= 1 and len(swrd2-swrd1) <= 1:
            return True
        return False

    
    def find_spell(self, word, curr_wrds):
        logging.debug('\tfind_spell({})'.format(word))
        for i, wrd in enumerate(curr_wrds):
            if self.is_spell(wrd,curr_wrds[i]) and wrd != word:
                logging.debug('\tfirst spell: {}'.format(curr_wrds[i]))
                return i
        return -1

    
    def correct(self, word, curr_wrds, curr_tags, word_next=None):

        if curr_tags[0] == keep:
            return word, False
        
        elif curr_tags[0] == '$REPLACE:INFLECTION':
            i = self.find_inflection(word, curr_wrds)
            if i>=0:
                return curr_wrds[i], False
            return word, False ### no change
        
        elif curr_tags[0] == '$REPLACE:SAMEPOS':
            i = self.find_samepos(word, curr_wrds)
            if i>=0:
                return curr_wrds[i], False
            return word, False ### no change
        
        elif curr_tags[0] == '$REPLACE:HOMOPHONE':
            i = self.find_homophone(word, curr_wrds)
            if i>=0:
                return curr_wrds[i], False
            return word, False ### no change
        
        elif curr_tags[0] == '$REPLACE:SPELL':
            i = self.find_spell(word, curr_wrds)
            if i>=0:
                return curr_wrds[i], False
            return word, False ### no change

        elif curr_tags[0] == '$APPEND':
            return word + " " + curr_wrds[0], False
        
        elif curr_tags[0] == '$DELETE':
            return "", False
        
        elif curr_tags[0] == '$MERGE':
            if word_next is not None:
                return word + word_next, True ### must delete next word
            else:
                return word, False
            
        elif curr_tags[0] == '$SWAP':
            if word_next is not None:
                return word_next + " " + word, True ### must delete next word
            else:
                return word, False
            
        elif curr_tags[0] == '$SPLIT':
            if word.startswith(curr_wrds[0]):
                k = len(curr_wrds[0])
                return word[:k] + " " + word[k:], False
            else:
                return word, False
            
        elif curr_tags[0] == '$CASE:FIRST':
            if word[0].isupper():
                word = word[0].lower() + word[1:]
            elif word[0].islower():
                word = word[0].upper() + word[1:]
            return word, False
        
        elif curr_tags[0] == '$CASE:UPPER':
            if word.islower():
                return word.upper(), False
            else:
                return word, False
            
        elif curr_tags[0] == '$CASE:LOWER':
            if word.isupper():
                return word.lower(), False
            else:
                return word, False
            
        elif curr_tags[0] == '$HYPHEN:SPLIT':
            if "-" in word:
                toks = word.split('-')
                return ' '.join(toks), False
            else:
                return word, False
            
        elif curr_tags[0] == '$HYPHEN:MERGE':
            if word_next is not None:
                return word + "-" + word_next, True ### must delete next word
            else:
                return word, False
            
        elif curr_tags[0].startswith('$INFLECT:'):
            newword = self.inflect(curr_tags[0][9:],word)
            if len(newword) > 0:
                return newword, False
            else:
                return word, False
        
        else:
            logging.error('Bad tag: '.format(curr_tags[0]))
            sys.exit()

    def inflect(self, tag0, word):
        logging.info('inflect({}, {})'.format(tag0,word))
        word_lc = word.lower()
        if word_lc not in self.lex.wrd2lem:
            logging.info('word: {} not found in lex'.format(word_lc))
            return word
        
        feats = tag0.split(';')
        logging.info('spacy feats: {}'.format(feats))
        pos = feats.pop(0)
        logging.info('spacy pos: {}'.format(pos))
        lems = self.lex.wrd2lem[word_lc]
        logging.info('lems: {}'.format(lems))

        for lem in lems:
            acceptable_words = self.lex.lempos2wrds[lem+separ+pos]
            logging.info('acceptable_words: {}'.format(acceptable_words))
            for feat in feats:
                logging.info('feat: {}'.format(feat))
                if lem+separ+pos+separ+feat in self.lex.lemposfeat2wrds:
                    reduced_words = self.lex.lemposfeat2wrds[lem+separ+pos+separ+feat]
                    logging.info('feat: {} reduced_words: {}'.format(feat,reduced_words))
                    acceptable_words = acceptable_words.intersection(reduced_words)
                    logging.info('feat: {} acceptable_words: {}'.format(feat,acceptable_words))
            if len(acceptable_words) == 1:
                return list(acceptable_words)[0]
        logging.info('no inflection found')
        return 'inflect(' + word + '|' + tag0 + ')'

