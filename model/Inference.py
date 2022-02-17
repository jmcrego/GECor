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
        model.eval()
        with torch.no_grad():
            dinputs = {}
            for inputs, indexs, idxs in testset:
                inputs = inputs.to(device)
                indexs = indexs.to(device)
                dinputs['input_ids'] = inputs
                outtag, outwrd = model(dinputs, indexs) ### forward


