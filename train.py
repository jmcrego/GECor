import sys
import os
import time
import random
import logging
import torch
import argparse
import torch.optim as optim
from GECor import GECor, load_or_create_checkpoint, load_checkpoint, save_checkpoint, CE2
from Learning import Learning
from Dataset import Dataset
from Vocab import Vocab
#from Tokenizer import Tokenizer
from FlaubertTok import FlaubertTok
from Utils import create_logger
from transformers import FlaubertModel, FlaubertTokenizer

######################################################################
### MAIN #############################################################
######################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Model file (required)', required=True)
    parser.add_argument('--train', help='Training data file (required)', required=True)
    parser.add_argument('--valid', help='Validation data file (required)')
    ### network
    parser.add_argument('--err', help='Vocabulary of errors (required)', required=True)
    parser.add_argument('--cor', help='Vocabulary of corrections', required=False)
    parser.add_argument('--n_subtokens', type=int, default=1, help='Number of word subtokens to predict', required=False)
    parser.add_argument('--aggregation', type=str, default="max", help='Aggregation when merging embeddings: first, last, max, avg, sum (max)')
    ### optim
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing value (0.1)')
    parser.add_argument('--loss', type=str, default="CE2", help='Loss function (CE2)')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta for CE2 loss (1.0)')
    parser.add_argument('--clip', type=float, default=0.0, help='Clip gradient norm of parameters (0.0)')
    parser.add_argument('--accum_n_batchs', type=int, default=4, help='Accumulate n batchs before model update (4)')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning Rate (0.00001)')
    ### learning
    parser.add_argument('--max_steps', type=int, default=0, help='Maximum training steps (0)')
    parser.add_argument('--max_epochs', type=int, default=0, help='Maximum training epochs (0)')
    parser.add_argument('--validate_every', type=int, default=5000, help='Validate every this steps (5000)')
    parser.add_argument('--save_every', type=int, default=10000, help='Save model every this steps (10000)')
    parser.add_argument('--report_every', type=int, default=100, help='Report every this steps (100)')
    parser.add_argument('--keep_last_n', type=int, default=2, help='Save last n models (2)')
    ### data
    parser.add_argument('--shard_size', type=int, default=2000000, help='Examples in shard (2000000)')
    parser.add_argument('--max_length', type=int, default=200, help='Maximum example length (200)')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size (4096)')
    parser.add_argument('--batch_type', type=str, default="tokens", help='Batch type: tokens or sentences (tokens)')
    ### others
    parser.add_argument('--cuda', action='store_true', help='Use cuda device instead of cpu')
    parser.add_argument('--seed', type=int, default=0, help='Seed for randomness (0)')
    parser.add_argument('--log_file', type=str, default="stderr", help='Log file (stderr)')
    parser.add_argument('--log', type=str, default="info", help='Log level (info)')
    args = parser.parse_args()

    create_logger(args.log_file,args.log)
    if args.seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    logging.info("Options = {}".format(args.__dict__))
    tic = time.time()
    
    ########################
    ### load model/optim ###
    ########################
    err = Vocab(args.err)
    cor = Vocab(args.cor) if args.cor is not None else None
    token = FlaubertTok("flaubert/flaubert_base_cased")
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    model = GECor(err, cor, encoder_name="flaubert/flaubert_base_cased", aggregation=args.aggregation, n_subtokens=args.n_subtokens).to(device)
    optim = optim.Adam(model.parameters(), lr=args.lr)
    last_step, model, optim = load_or_create_checkpoint(args.model, model, optim, device)
    
    ############################
    ### build scheduler/loss ###
    ############################
    if args.loss == 'CE2':
        criter = CE2(args.label_smoothing,args.beta).to(device)
    else:
        logging.error('Invalid --loss option')

    #############
    ### learn ###
    #############
    validset = Dataset(args.valid, err, cor, token, args) if args.valid is not None else None
    trainset = Dataset(args.train, err, cor, token, args)
    learning = Learning(model, optim, criter, last_step, trainset, validset, cor, err, args, device)
    
    toc = time.time()
    logging.info('Done ({:.2f} seconds)'.format(toc-tic))










    
