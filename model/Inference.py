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

    def __init__(self, model, testset, words, tags, idx_PAD, args, device):
        super(Inference, self).__init__()
        self.tags = tags
        self.words = words
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
        logging.info('idx: {}'.format(idx))
        corrected_words = []
        for i in range(1,len(words)-1):
            if words[i] == "": ### has been deleted
                continue
            curr_tags = [self.tags[tag_idx] for tag_idx in kbest_tags[i].tolist()]
            curr_wrds = [self.words[wrd_idx] for wrd_idx in kbest_wrds[i].tolist()]
            logging.info("{}\t{}\t{}".format(words[i], curr_tags, curr_wrds))
            corrected_word, delete_next = self.correct(words[i], curr_wrds,curr_tags, word_next=words[i+1] if i+1<len(words) else None)
            corrected_words.append(corrected_word)
            if delete_next:
                words[i+1] = ""
        self.corrected_sentences[idx] = corrected_words

    def correct(self, word, curr_wrds, curr_tags, word_next=None):
        if curr_tags[0] == keep:
            return word, False
        elif curr_tags[0] == '$REPLACE:INFLECTION':
            return curr_wrds[0], False
        elif curr_tags[0] == '$REPLACE:SAMEPOS':
            return curr_wrds[0], False
        elif curr_tags[0] == '$REPLACE:HOMOPHONE':
            return curr_wrds[0], False
        elif curr_tags[0] == '$REPLACE:SPELL':
            return curr_wrds[0], False
        elif curr_tags[0] == '$APPEND':
            return word + " " + curr_wrds[0], False
        elif curr_tags[0] == '$DELETE':
            return "", False
        elif curr_tags[0] == '$MERGE': ### must delete next word
            if word_next is not None:
                return word + word_next, True
        elif curr_tags[0] == '$SWAP': ### must delete next word
            if word_next is not None:
                return word_next + " " + word, True
        elif curr_tags[0] == '$SPLIT':
            if word.startswith(curr_wrds[0]):
                k = len(curr_wrds[0])
                return word + " " + curr_wrds[0][k:], False
        elif curr_tags[0] == '$CASE:FIRST':
            if word[0].isupper():
                word[0] = word[0].lower()
            elif word[0].islower():
                word[0] = word[0].upper()
            return word, False
        elif curr_tags[0] == '$CASE:UPPER':
            if word.is_lower():
                return word.upper(), False
        elif curr_tags[0] == '$CASE:LOWER':
            if word.is_upper():
                return word.lower(), False
        elif curr_tags[0] == '$HYPHEN:SPLIT':
            if "-" in word:
                toks = word.split('-')
                return ' '.join(toks), False
        elif curr_tags[0] == '$HYPHEN:MERGE': ### must delete next word
            if word_next is not None:
                return word + "-" + word_next, True
        elif curr_tags[0].startswith('$INFLECT:'):
            return curr_tags[0] + "(" + word + ")", False
        else:
            logging.error('Bad tag: '.format(curr_tags[0]))
            sys.exit()
