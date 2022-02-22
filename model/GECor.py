import os
import sys
import glob
import torch
import logging
import torch_scatter
import torch.nn as nn
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
    logging.info('Loaded checkpoint step={} encoder_freezed={} from {} device={}'.format(step,model.encoder_freezed,fmodel,device))
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
    logging.info('Saved checkpoint step={} encoder_freezed={} in {}.{:08d}.pt'.format(step,model.encoder_freezed,fmodel,step))
    files = sorted(glob.glob(fmodel + '.????????.pt')) 
    while keep_last_n > 0 and len(files) > keep_last_n:
        f = files.pop(0)
        os.remove(f) ### first is the oldest
        logging.debug('Removed checkpoint {}'.format(f))

    
class CE2(torch.nn.Module):

    def __init__(self, label_smoothing=0.0, beta=1.0):
        super(CE2, self).__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing,reduction='mean')
        self.beta = beta

    def forward(self, outtag, outwrd, msktag, mskwrd, reftag, refwrd):
        loss = self.ce(outtag[msktag.bool()], reftag[msktag.bool()]) + self.beta * self.ce(outwrd[mskwrd.bool()], refwrd[mskwrd.bool()])
        return loss    

    
class GECor(nn.Module):

    def __init__(self, n_tags, n_words, encoder_name="flaubert/flaubert_base_cased", aggregation='sum', tag_embedding=0, encoder_freezed=True):
        super(GECor, self).__init__()
        self.encoder = FlaubertModel.from_pretrained(encoder_name)
        self.emb_size = self.encoder.attentions[0].out_lin.out_features
        self.linear_layer_tags = nn.Linear(self.emb_size, n_tags)
        self.linear_layer_words = nn.Linear(self.emb_size+tag_embedding, n_words)
        if tag_embedding > 0:
            self.tag_embedding = nn.Embedding(n_tags, tag_embedding, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, device=None, dtype=None)
        self.aggregation = aggregation
        self.n_tags = n_tags
        self.n_words = n_words
        self.encoder_freezed = encoder_freezed

    def forward(self, inputs, indexs):
        #words : La|· maison|· blancHe|$SPELL_blanche
        #bpe   : <s> La mai son blan cHe </s> <pad>*
        #inputs: 0   3  9   5   6    7   1    2
        #indexs: 0   1  2   2   3    3   4    5
        if self.encoder_freezed:
            with torch.no_grad():
                embeddings = self.encoder(**inputs)
        else:
            embeddings = self.encoder(**inputs) 
        embeddings = embeddings.last_hidden_state

        if torch.max(indexs) > embeddings.shape[1]-1:
            logging.error('Indexs bad formatted!')
            sys.exit()

        embeddings_merge = torch.zeros_like(embeddings, dtype=embeddings.dtype, device=embeddings.device)
        torch_scatter.segment_coo(embeddings, indexs, out=embeddings_merge, reduce=self.aggregation)
        out_tags_merge = self.linear_layer_tags(embeddings_merge)
#        logging.info('out_tags_merge.shape=',out_tags_merge.shape)
        
        tags_idx = torch.argsort(out_tags_merge, dim=-1, descending=True)[:,:,0] #[bs, l, ts] => [bs, l, 1] ### get the one-best of each token
#        logging.info('tags_idx.shape=',tags_idx.shape)
        embeddings_tags = self.tag_embedding(tags_idx)
#        logging.info('embeddings_tags.shape=',embeddings_tags.shape)
        
        embeddings_cat = torch.cat((embeddings_merge, embeddings_tags),dim=-1) #[bs, l, es+ts]
#        logging.info('embeddings_cat.shape=',embeddings_cat.shape)
        out_words_merge = self.linear_layer_words(embeddings_cat) #[bs, l, ws]
#        logging.info('out_words_merge.shape=',out_words_merge.shape)
        return out_tags_merge, out_words_merge

    def parameters(self):
        if self.encoder_freezed:
            logging.info('Optimizer with freezed encoder')
            return iter(itertools.chain(self.linear_layer_tags.parameters(), self.linear_layer_words.parameters()))
        logging.info('Optimizer with unfreezed encoder')
        return super().parameters()    
    
if __name__ == '__main__':

    #src = torch.tensor(  [[1.0, 1.0, 1.0, 0.0], [2.0, 2.0, 2.0, 0.0]])
    src = torch.randn(2, 4, 3)
    index = torch.tensor([[0,   1,   1,   2],   [0,   0,   1,   2]])

    out = torch.zeros_like(src)
    torch_scatter.segment_coo(src, index, out=out, reduce="sum")
    print('src',src)
    print('index',index)
    print('out',out)
    
