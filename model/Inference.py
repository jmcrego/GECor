# -*- coding: utf-8 -*-

import sys
import os
import logging
import numpy as np
import torch
import time
from model.GECor import save_checkpoint
from collections import defaultdict

class Inference():

    def __init__(self, model, testset, words, tags, idx_PAD, args, device):
        super(Inference, self).__init__()
        self.tags = tags
        self.words = words
        self.args = args
        model.eval()
        with torch.no_grad():
            dinputs = {}
            for inputs, indexs, words, idxs in testset:
                inputs = inputs.to(device)
                indexs = indexs.to(device)
                dinputs['input_ids'] = inputs
                outtag, outwrd = model(dinputs, indexs) ### [bs, l, ts], [bs, l, ws]
                kbest_tags = outtag.argsort(dim=2, descending=True)[:,:,:self.args.K] #[bs, l, K]
                kbest_wrds = outwrd.argsort(dim=2, descending=True)[:,:,:self.args.K] #[bs, l, K]
                for i in range(kbest_tags.shape[0]):
                    self.sentence(idxs[i],words[i],kbest_tags[i],kbest_wrds[i])

                    
    def sentence(self,idx,words,kbest_tags,kbest_wrds):
        #kbest_tags [l,K]
        #kbest_wrds [l,K]
        logging.info('idx: {}'.format(idx))
        corrected_words = []
        for i in range(1,len(words)-1):
            curr_tags = [self.tags[tag_idx] for tag_idx in kbest_tags[i].tolist()]
            curr_wrds = [self.words[wrd_idx] for wrd_idx in kbest_wrds[i].tolist()]
            logging.info("{}\t{}\t{}".format(words[i], ' '.join(curr_tags), ' '.join(curr_wrds)))
        return idx, corrected_words
