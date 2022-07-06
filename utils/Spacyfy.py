import sys
import time
import spacy
import logging
from utils.Lexicon import Lexicon
from model import Vocab
from noise.Homophones import Homophones
from utils.FlaubertTok import FlaubertTok
from utils.Utils import shape_of_word

class Spacyfy():

    def __init__(self, m, b, n, only_tokenize, flex, voc_sha=None):
        self.b = b #batch size
        self.n = n #n preproces
        self.only_tokenize = only_tokenize
        self.lines = []
        self.lex = Lexicon(flex)
        self.tok = FlaubertTok("flaubert/flaubert_base_cased")
        self.vocS = Vocab(voc_sha) if voc_sha is not None else None
            
        if only_tokenize:
            from spacy.lang.fr import French
            nlp = French()
            self.nlp = nlp.tokenizer
            self.n = 1
            logging.info('Loaded French tokenizer for tokenization')
        else:
            self.nlp = spacy.load(m, exclude=["parser", "ner", "attribute_ruler"]) # ["tok2vec", "morphologizer", "lemmatizer", "tagger", "parser", "ner", "attribute_ruler"]
            logging.info('Loaded {} with modules {}'.format(m, self.nlp.pipe_names))

    def read_lines(self,fn):
        logging.info('Reading from {}...'.format(fn))
        with open(fn, 'r', encoding='utf-8') if fn != "stdin" else sys.stdin as fd:    
            lines = fd.readlines()
        self.lines = [s.strip() for s in lines]
        logging.info('Read {} lines from {}'.format(len(self.lines),fn))

    def __len__(self):
        return len(self.lines)
    
    def __iter__(self):
        if len(self.lines) == 0:
            logging.warning('No lines available to spacyfy')
            return
        ### lines to spacyfy are already stored in self.lines
        ### split list of strings into batchs (list of list of strings)
        ### each batch contains up to n*b lines
        bs = self.b * self.n
        batchs = [self.lines[i:min(i+bs,len(self.lines))] for i in range(0, len(self.lines), bs)]
        for batch in batchs:
            #each of the n processes has a batch with b lines (batchs built internally)
            docs = list(self.nlp.pipe(batch, batch_size=self.b))  if self.only_tokenize else list(self.nlp.pipe(batch, n_process=self.n, batch_size=self.b))
            #prepare output list of list of dicts (words)
            for doc in docs:
                line= []
                for token in doc:
                    if ' ' in token.text or ' ' in token.lemma_:
                        continue
                    line.append(self.token2dword(token))
                yield line
                    
            #lines = [[self.token2dword(token) for token in doc] for doc in docs]
            #for line in lines:
            #    yield line
    
    def token2dword(self,token):
        raw = token.text
        shape = shape_of_word(raw)
        ids = self.tok.ids(raw, is_split_into_words=True)
        txt = self.lex.inlexicon(raw)
        d = {'r':raw, 's':shape, 'i': ids, 't': txt}
        if self.vocS is not None:
            d['is'] = self.vocS[shape]
        if self.only_tokenize or txt == '':
            return d
        plm = self.lex.spacy2morphalou(txt, str(token.lemma_), str(token.pos_), str(token.morph)) #this is used to generate noise
        if plm == '':
            return d
        d['plm'] = plm
        return d

