import sys
import os
import time
import random
import logging
import torch
import argparse
import torch.optim as optim
from model.GECor import GECor, load_or_create_checkpoint, load_checkpoint, save_checkpoint, CE2, load_model
from model.Inference import Inference
from model.Dataset import Dataset
from model.Vocab import Vocab
from model.Tokenizer import Tokenizer
from model.Utils import create_logger
from transformers import FlaubertModel, FlaubertTokenizer

######################################################################
### MAIN #############################################################
######################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Model file (required)', required=True)
    parser.add_argument('--test', help='Testing data file (required)', required=True)
    ### network
    parser.add_argument('--tags', help='Vocabulary of tags (required)', required=True)
    parser.add_argument('--words', help='Vocabulary of words (required)', required=True)
    parser.add_argument('--aggregation', type=str, default="max", help='Aggregation when merging embeddings (max)')
    ### 
    parser.add_argument('--Kt', type=int, default=1, help='Output K-best tag options (1)')    
    parser.add_argument('--Kw', type=int, default=1, help='Output K-best word options (1)')    
    ### data
    parser.add_argument('--shard_size', type=int, default=5000000, help='Examples in shard (5000000)')
    parser.add_argument('--max_length', type=int, default=0, help='Maximum example length (0)')    
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size (4096)')
    parser.add_argument('--batch_type', type=str, default="tokens", help='Batch type: tokens or sentences (tokens)')
    ### others
    parser.add_argument('--cuda', action='store_true', help='Use cuda device instead of cpu')
    parser.add_argument('--seed', type=int, default=0, help='Seed for randomness (0)')
    parser.add_argument('--log_file', type=str, default="stderr", help='Log file (stderr)')
    parser.add_argument('--log_level', type=str, default="info", help='Log level (info)')
    args = parser.parse_args()

    create_logger(args.log_file,args.log_level)
    if args.seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    logging.info("Options = {}".format(args.__dict__))
    tic = time.time()

    ########################
    ### load model/optim ###
    ########################
    tags = Vocab(args.tags)
    words = Vocab(args.words)
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    model = GECor(len(tags), len(words), encoder_name="flaubert/flaubert_base_cased").to(device)
    model = load_model(args.model,model,device)

    #############
    ### infer ###
    #############
    tokenizer = Tokenizer("flaubert/flaubert_base_cased")
    testset = Dataset(args.test, tags, words, tokenizer, args)
    inference = Inference(model, testset, words, tags, words.idx_PAD, args, device)

    toc = time.time()
    logging.info('Done ({:.2f} seconds)'.format(toc-tic))










    
