import os
import sys
import glob
import torch
import logging
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F
from transformers import FlaubertModel
import itertools

def load_or_create_checkpoint(fmodel, model, optimizer, device):
    files = sorted(glob.glob("{}.????????.pt".format(fmodel)))
    if len(files) == 0:
        step = 0
        save_checkpoint(fmodel, model, optimizer, step, 0)
    else:
        step, model, optimizer = load_checkpoint(fmodel, model, optimizer, device)
    return step, model, optimizer


def load_checkpoint(fmodel, model, optimizer, device):
    step = 0
    files = sorted(glob.glob("{}.????????.pt".format(fmodel)))
    if len(files) == 0:
        logging.info('No model found')
        sys.exit()
    file = files[-1] ### last is the newest
    checkpoint = torch.load(file, map_location=device)
    step = checkpoint['step']
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    logging.info('Loaded checkpoint step={} from {} device={}'.format(step,fmodel,device))
    return step, model, optimizer

def load_model(fmodel, model, device):
    checkpoint = torch.load(fmodel, map_location=device)
    model.load_state_dict(checkpoint['model'])
    return model
    
def save_checkpoint(fmodel, model, optimizer, step, keep_last_n):
    if os.path.isfile("{}.{:08d}.pt".format(fmodel,step)):
        logging.info('Checkpoint already exists')
        return
    checkpoint = { 'step': step, 'model': model.state_dict(), 'optimizer': optimizer.state_dict() }
    torch.save(checkpoint, "{}.{:08d}.pt".format(fmodel,step))
    logging.info('Saved checkpoint step={} in {}.{:08d}.pt'.format(step,fmodel,step))
    files = sorted(glob.glob(fmodel + '.????????.pt')) 
    while keep_last_n > 0 and len(files) > keep_last_n:
        f = files.pop(0)
        os.remove(f) ### first is the oldest
        logging.debug('Removed checkpoint {}'.format(f))

    
class CE2(torch.nn.Module):

    def __init__(self, label_smoothing=0.0, beta=1.0):
        super(CE2, self).__init__()
        self.crossent = nn.CrossEntropyLoss(label_smoothing=label_smoothing,reduction='mean') #only tokens not padded are used to compute loss
        self.beta = beta

    def forward(self, outtag, outcor, msktag, mskcor, reftag, refcor):
        #loss_tag = self.crossent(outtag[msktag.bool()], reftag[msktag.bool()])
        #loss_cor = self.crossent(outcor[mskcor.bool()], refcor[mskcor.bool()])
        #return loss_tag + self.beta * loss_cor
        (bs, lt, ts) = outtag.shape
        (_,  lc, cs) = outcor.shape
        #logging.info('outtag.shape = {}'.format(outtag.shape)) #[bs, lt, ts]
        #logging.info('outcor.shape = {}'.format(outcor.shape)) #[bs, lc, cs]
        #logging.info('msktag.shape = {}'.format(msktag.shape)) #[bs, lt]
        #logging.info('mskcor.shape = {}'.format(mskcor.shape)) #[bs, lc, 1]
        #logging.info('reftag.shape = {}'.format(reftag.shape)) #[bs, lt]
        #logging.info('refcor.shape = {}'.format(refcor.shape)) #[bs, lc, 1]
        outtag = outtag.reshape(bs*lt,-1) #[bs*lt,ts]
        outcor = outcor.reshape(bs*lc,-1) #[bs*lc,cs]
        msktag = msktag.reshape(bs*lt) #[bs*lt]
        mskcor = mskcor.reshape(bs*lc) #[bs*lc]
        reftag = reftag.reshape(bs*lt) #[bs*lt]
        refcor = refcor.reshape(bs*lc) #[bs*lc]

        outtag_mask = outtag[msktag.bool()] #[N,ts]
        reftag_mask = reftag[msktag.bool()] #[N]
        loss_tag = self.crossent(outtag_mask, reftag_mask)
        #logging.info('loss_tag={} : {} out of {} elements'.format(loss_tag.item(), msktag.sum(), torch.numel(msktag)))

        outcor_mask = outcor[mskcor.bool()] #[M,cs]
        refcor_mask = refcor[mskcor.bool()] #[M]
        loss_cor = self.crossent(outcor_mask, refcor_mask) 
        #logging.info('loss_cor={} : {} out of {} elements'.format(loss_cor.item(), mskcor.sum(), torch.numel(mskcor)))

        loss = loss_tag + self.beta * loss_cor
        logging.debug('CE2 loss: {:.3f} + {:.3f} = {:.3f}'.format(loss_tag.item(),loss_cor.item(),loss.item()))
        return loss    

    
class GECor(nn.Module):

    def __init__(self, tags, cors, encoder_name="flaubert/flaubert_base_cased", aggregation='sum', n_subtokens=1):
        super(GECor, self).__init__() #flaubert_base_cased info in https://huggingface.co/flaubert/flaubert_base_cased/tree/main can be accessed via self.encoder.config.vocab_size

        self.encoder = FlaubertModel.from_pretrained(encoder_name)
        self.n_tags = len(tags)
        self.idx_PAD_tag = tags.idx_PAD
        if cors is None:
            self.n_cors = self.encoder.config.vocab_size #68729 #https://huggingface.co/flaubert/flaubert_base_cased/tree/main can be accessed via self.encoder.config.vocab_size
            self.idx_PAD_cor = 2 #"<pad>": 2, read in https://huggingface.co/flaubert/flaubert_base_cased/raw/main/vocab.json
        else:
            n_subtokens = 1
            self.idx_PAD_cor = cors.idx_PAD
            self.n_cors = len(cors)
        
        self.aggregation = aggregation
        self.n_subtokens = n_subtokens            
        self.emb_size = self.encoder.config.emb_dim
        self.linear_layer_tags = nn.Linear(self.emb_size, self.n_tags)
        #self.n_linear_layer_cors = nn.ModuleList([nn.Linear(self.emb_size, self.n_cors) for i in range(self.n_subtokens)])
        self.linear_layer_cors = nn.Linear(self.emb_size, self.n_cors*self.n_subtokens)
        
    def forward(self, inputs, indexs):
        #####################
        ### encoder layer ###
        #####################
        embeddings = self.encoder(**inputs).last_hidden_state #[bs, l, es]
        if torch.max(indexs) > embeddings.shape[1]-1:
            logging.error('Indexs bad formatted!')
            sys.exit()

        ###################
        ### aggregation ###
        ###################
        if self.aggregation in ['sum', 'avg', 'max']:
            embeddings_aggregate = torch.zeros_like(embeddings, dtype=embeddings.dtype, device=embeddings.device)
            torch_scatter.segment_coo(embeddings, indexs, out=embeddings_aggregate, reduce=self.aggregation)
        elif self.aggregation in ['first','last']:
            embeddings_aggregate = embeddings
        else:
            logging.error('Bad aggregation value: {}'.format(self.aggregation))
            sys.exit()

        ##################
        ### tags layer ###
        ##################
        out_tags = self.linear_layer_tags(embeddings_aggregate) #[bs, l, ts]

        ####################
        ### cors layer/s ###
        ####################
        #for i in range(len(self.n_linear_layer_cors)): ### i should concat cs rather than building a list
        #    if i==0:
        #        out_cors = self.n_linear_layer_cors[i](embeddings_aggregate) #[bs, l, cs]
        #    else:
        #        out_cors = torch.cat((out_cors, self.n_linear_layer_cors[i](embeddings_aggregate)), dim=-1) #[bs, l, cs*i] 
        out_cors = self.linear_layer_cors(embeddings_aggregate) #[bs, l, cs*n_subtokens]

        return out_tags, out_cors

    def parameters(self):
        return super().parameters()    
    
    
