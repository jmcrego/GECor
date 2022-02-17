# -*- coding: utf-8 -*-

import sys
import os
import logging
import numpy as np
import torch
import time
import torch.optim as optim
from model.GECor import save_checkpoint
from collections import defaultdict

try:
    from torch.utils.tensorboard import SummaryWriter
    tensorboard = True
except ImportError:
    tensorboard = False


class Score():
    
    def __init__(self, tags, writer):
        self.writer = writer
        self.tags = tags
        self.tag2tp = [0] * len(tags)
        self.tag2fp = [0] * len(tags)
        self.tag2fn = [0] * len(tags)
        self.tag2n = [0] * len(tags)
        self.tag_ok = 0
        self.tag_n = 0
        self.wrd_ok = 0
        self.wrd_n = 0
        self.Loss = 0.0
        self.nsteps = 0
        self.start = time.time()

    def add(self, outtag, outwrd, msktag, mskwrd, reftag, refwrd, loss):
        self.Loss += loss #averaged per toks in batch
        self.nsteps += 1
        outtag = torch.argsort(outtag, dim=-1, descending=True)[:,:,0] #[bs, l, ts] => [bs, l, 1] ### get the one-best of each token
        outtag = outtag.reshape(-1) #[bs*l]
        reftag = reftag.reshape(-1) #[bs*l]
        msktag = msktag.reshape(-1) #[bs*l]        
        outtag = outtag[msktag == 1]
        reftag = reftag[msktag == 1]
        assert(outtag.shape == reftag.shape)
        self.tag_n += outtag.shape[0]
        self.tag_ok += torch.sum(reftag == outtag)
        reftag = reftag.tolist()
        outtag = outtag.tolist()
        for i in range(len(reftag)):
            self.tag2n[reftag[i]] += 1
            if reftag[i] == outtag[i]:
                self.tag2tp[reftag[i]] +=1
            else:
                self.tag2fp[outtag[i]] +=1
                self.tag2fn[reftag[i]] +=1
        outwrd = torch.argsort(outwrd, dim=-1, descending=True)[:,:,0] #[bs, l, ts] => [bs, l, 1] ### get the one-best of each token
        outwrd = outwrd.reshape(-1) #[bs*l]
        refwrd = refwrd.reshape(-1) #[bs*l]
        mskwrd = mskwrd.reshape(-1) #[bs*l]        
        outwrd = outwrd[mskwrd == 1]
        refwrd = refwrd[mskwrd == 1]
        assert(outwrd.shape == refwrd.shape)
        self.wrd_n += outwrd.shape[0]
        self.wrd_ok += torch.sum(refwrd == outwrd)

    def report(self, step, trnval):
        end = time.time()
        steps_per_sec = (self.nsteps) / (end - self.start)
        loss_per_tok = self.Loss / self.nsteps
        logging.info("{}/Loss:{:.6f} step:{} steps/sec:{:.2f}".format(trnval, loss_per_tok, step, steps_per_sec))
        if self.writer is not None:
            self.writer.add_scalar('Loss/{}'.format(trnval), loss_per_tok, step)
        if trnval == 'valid':
            logging.info('{}/Acc: [tags] {}/{} ({:.3f}) [words] {}/{} ({:.3f})'.format(trnval,self.tag_ok,self.tag_n,1.0*self.tag_ok/self.tag_n,self.wrd_ok,self.wrd_n,1.0*self.wrd_ok/self.wrd_n))
            for tag in range(len(self.tag2n)):
                tp = self.tag2tp[tag]
                fp = self.tag2fp[tag]
                fn = self.tag2fn[tag]
                #logging.info('valid/TP: {:.3f} ({}) [{}]'.format(tp,self.tag2n[tag],self.tags[tag]))
                if tp>0:
                    P = tp / (tp + fp)
                    R = tp / (tp + fn)
                    F1 = 2.0*P*R / (P + R)
                    F2 = 5.0*P*R / (4*P + R)
                    logging.info('valid/P|R|F1|F2: {:.3f}|{:.3f}|{:.3f}|{:.3f} ({}) [{}]'.format(P,R,F1,F2,self.tag2n[tag],self.tags[tag]))
            

class Learning():

    def __init__(self, model, optim, criter, step, trainset, validset, words, tags, idx_PAD, args, device):
        super(Learning, self).__init__()
        writer = SummaryWriter(log_dir=args.model, comment='', purge_step=None, max_queue=10, flush_secs=60, filename_suffix='') if tensorboard else None
        n_epoch = 0
        optim.zero_grad() # sets gradients to zero
        score = Score(tags, writer)
        while True: #repeat epochs
            n_epoch += 1
            logging.info('Epoch {}'.format(n_epoch))
            n_batch = 0
            loss_accum = 0.0
            dinputs = {}
            for inputs, indexs, reftag, refwrd in trainset:
                inputs = inputs.to(device)
                indexs = indexs.to(device)
                reftag = reftag.to(device)
                refwrd = refwrd.to(device)
                #source: La|· maison|· blancHe|$SPELL_blanche
                #words : <s>  La  mai son blan cHe </s> <pad>*
                #inputs: 0    37  92  25  36   987 1    2*
                #indexs: 0    1   2   2   3    3   4    5*
                #reftag: PAD  ·   SPELL   PAD PAD  PAD  PAD*
                #refwrd: PAD  PAD PAD     blanche  PAD  PAD*
                model.train()
                criter.train()
                n_batch += 1
                dinputs['input_ids'] = inputs
                if args.unfreeze_encoder and model.encoder_freezed and step == args.unfreeze_encoder: ### initialize optimizer with all parameters
                    model.encoder_freezed = False
                    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
                outtag, outwrd = model(dinputs, indexs) #forward, no log_softmax is applied
                msktag = torch.ones_like(reftag).to(outtag.device)
                mskwrd = torch.ones_like(refwrd).to(outwrd.device)
                msktag[reftag == idx_PAD] = 0
                mskwrd[refwrd == idx_PAD] = 0
                loss = criter(outtag, outwrd, msktag, mskwrd, reftag, refwrd) / float(args.accum_n_batchs) #average of losses in batch (already normalized by tokens in batch) (n batchs will be accumulated before model update, so i normalize by n batchs)
                loss.backward()
                loss_accum += loss
                if n_batch % args.accum_n_batchs == 0:
                    step += 1 ### current step
                    ### optimize ###
                    score.add(outtag, outwrd, msktag, mskwrd, reftag, refwrd, loss_accum.item())
                    if args.clip > 0.0: # clip gradients norm
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    optim.step() # updates model parameters
                    optim.zero_grad() # sets gradients to zero for next update
                    loss_accum = 0.0
                    ### report ###
                    if args.report_every and step % args.report_every == 0:
                        score.report(step, 'train')
                        score = Score(tags, writer)
                    ### validate ###
                    if args.validate_every and step % args.validate_every == 0: 
                        self.validate(model, criter, step, validset, tags, idx_PAD, writer, device)
                    ### save ###
                    if args.save_every and step % args.save_every == 0: 
                        save_checkpoint(args.model, model, optim, step, args.keep_last_n)
                    ### stop by max_steps ###
                    if args.max_steps and step >= args.max_steps: 
                        self.validate(model, criter, step, validset, tags, idx_PAD, writer, device)
                        save_checkpoint(args.model, model, optim, step, args.keep_last_n)
                        logging.info('Learning STOP by [steps={}]'.format(step))
                        return
            ### stop by max_epochs ###
            if args.max_epochs and n_epoch >= args.max_epochs:
                self.validate(model, criter, step, validset, tags, idx_PAD, writer, device)
                save_checkpoint(args.model, model, optim, step, args.keep_last_n)
                logging.info('Learning STOP by [epochs={}]'.format(n_epoch))
                return

    def validate(self, model, criter, step, validset, tags, idx_PAD, writer, device):
        if validset is None:
            return
        model.eval()
        criter.eval()
        score = Score(tags, writer)
        with torch.no_grad():
            dinputs = {}
            for inputs, indexs, reftag, refwrd in validset:
                inputs = inputs.to(device)
                indexs = indexs.to(device)
                reftag = reftag.to(device)
                refwrd = refwrd.to(device)
                dinputs['input_ids'] = inputs
                outtag, outwrd = model(dinputs, indexs) ### forward
                msktag = torch.ones_like(reftag).to(outtag.device)
                mskwrd = torch.ones_like(refwrd).to(outwrd.device)
                msktag[reftag == idx_PAD] = 0
                mskwrd[refwrd == idx_PAD] = 0
                loss = criter(outtag, outwrd, msktag, mskwrd, reftag, refwrd)
                score.add(outtag, outwrd, msktag, mskwrd, reftag, refwrd, loss.item())
        score.report(step,'valid')

