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

    def __init__(self, m, b, n, only_tokenize, flex, shapes=None, inlexi=None):
        self.b = b #batch size
        self.n = n #n preproces
        self.only_tokenize = only_tokenize
        self.lines = []
        self.lex = Lexicon(flex)
        self.tok = FlaubertTok("flaubert/flaubert_base_cased")
        self.vocS = Vocab(shapes) if shapes is not None else None
        self.vocI = Vocab(inlexi) if inlexi is not None else None
            
        if only_tokenize:
            from spacy.lang.fr import French
            nlp = French()
            self.nlp = nlp.tokenizer
            self.n = 1
            logging.info('Loaded French tokenizer for tokenization')
        else:
            self.nlp = spacy.load(m, exclude=["parser", "ner", "attribute_ruler"]) # ["tok2vec", "morphologizer", "lemmatizer", "tagger", "parser", "ner", "attribute_ruler"]
            logging.info('Loaded {} with modules {}'.format(m, self.nlp.pipe_names))

    def read_lines(self,fn,one_out_of=1):
        logging.info('Reading from {}... (using 1 out of {} examples)'.format(fn,one_out_of))
        with open(fn, 'r', encoding='utf-8') if fn != "stdin" else sys.stdin as fd:
            self.lines = []
            for i,line in enumerate(fd):
                line2 = line.replace(u'\xa0',' ').replace(u'\u202f',' ').replace(u'\u00ad','').replace(u'\u200f','').replace(u'\u2009',' ')
                if line != line2:
                    logging.debug('removed characters in line:{}\n{}{}'.format(i+1,line,line2))
                    line = line2
                if i%one_out_of == 0:
                    self.lines.append(line.strip())
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
                    
    def token2dword(self,token):
        raw = token.text
        shape = shape_of_word(raw)
        ids = self.tok.ids(raw, is_split_into_words=True)
        txt = self.lex.inlexicon(raw)
        inlex = txt is not None #True or False
        d = {'raw':raw, 'shp':shape, 'iraw': ids, 'lex': txt}
        if self.vocS is not None:
            d['ishp'] = self.vocS[shape]
        if self.vocI is not None:
            d['ilex'] = self.vocI[str(inlex)]
        if self.only_tokenize or txt is None:
            return d
        plm = self.lex.spacy2morphalou(txt, str(token.lemma_), str(token.pos_), str(token.morph)) #this is used to generate noise
        if plm is not None:
            d['plm'] = plm
        return d

