import sys
import logging
from transformers import FlaubertTokenizer

class FlaubertTok():

    def __init__(self, model="flaubert/flaubert_base_cased", do_lowercase=False, max_ids_len=500):
        self.max_ids_len = max_ids_len
        self.flauberttok = FlaubertTokenizer.from_pretrained(model, do_lowercase=do_lowercase)
        self.idx_BOS = 0
        self.idx_EOS = 1
        self.idx_PAD = 2
        self.str_BOS = '<s>'
        self.str_EOS = '</s>'
        self.str_PAD = '<pad>'
        logging.info('Loaded FlaubertTok')

    def ids(self, l, is_split_into_words=False, add_special_tokens=False):
        l = l.rstrip()
        lids = self.flauberttok(l, is_split_into_words=is_split_into_words, add_special_tokens=add_special_tokens)['input_ids']
        if len(lids) > self.max_ids_len:
            last_ids = self.max_ids_len-1
            while last_ids > 0 and not self.flauberttok.convert_ids_to_tokens(lids[last_ids]).endswith('</w>'):
                last_ids -= 1
            lids = lids[:last_ids+1]        
        return lids
        
    def pretok(self, l, is_split_into_words=False, add_special_tokens=False):
        #######l = C'est mon ultim exemple.
        wrds = [] #C' est mon ultim exemple .
        lids = [] #[[113], [27], [151], [47981, 356], [305], [16]]
        curr_wrd = []
        curr_ids = []
        for ids in self.ids(l, is_split_into_words=is_split_into_words, add_special_tokens=add_special_tokens):
            str_ids = self.flauberttok.convert_ids_to_tokens(ids)
            curr_wrd.append(str_ids)
            curr_ids.append(ids)
            if str_ids.endswith('</w>') or str_ids == '<unk>' or str_ids == '<s>' or str_ids == '</s>':
                wrds.append(''.join(curr_wrd).replace('</w>',''))
                lids.append(curr_ids)
                curr_wrd = []
                curr_ids = []                
        if len(curr_wrd):
            sys.stderr.write('warning: bad finished sentence, wrd:{} ids:{} from sentence {} into words {}\n'.format(curr_wrd, curr_ids, l, wrds))
        assert(len(wrds) == len(lids))
        return wrds, lids

    def subtok(self, l, is_split_into_words=False, add_special_tokens=False):
        ##########l = C'est mon ultim exemple.
        subwrds = [] #C' est mon ulti m exemple .
        lids = []    #[113, 27, 151, 47981, 356, 305, 16]
        for ids in self.ids(l, is_split_into_words=is_split_into_words, add_special_tokens=add_special_tokens):
            str_ids = self.flauberttok.convert_ids_to_tokens(ids)
            subwrds.append(str_ids.replace('</w>',''))
            lids.append(ids)
        assert(len(subwrds) == len(lids))
        return subwrds, lids

    def words_ids2words_subwords_lids(self, ids):
        if len(ids) == 0:
            logging.warning("warning empty ids: {}".format(ids))
            return [], [], [], []

        #ids [int] : list of ints
        #Returns:
        # words     [string] : list of strings
        # ids2words [int] : list of ints
        # subwords  [string] : list of strings
        # lids      [[int]] : list of subtokens(ids) of each token

        subwords = self.flauberttok.convert_ids_to_tokens(ids)
        assert(len(ids) == len(subwords))
        ids2words = []
        lids = []
        words = [''] #first word prepared                                                                                                                                                                                                                                                           
        for i in range(len(subwords)):
            if subwords[i] == '<s>':
                words[-1] = subwords[i]
                ids2words.append(len(words)-1)
                if i < len(subwords)-1:
                    words.append('') ### word finished prepare new
                    
            elif subwords[i] == '</s>':
                words[-1] = subwords[i]
                ids2words.append(len(words)-1)
                if i < len(subwords)-1:
                    words.append('') ### word finished prepare new

            elif subwords[i].endswith('</w>'):
                subwords[i] = subwords[i][:-4]
                words[-1] += subwords[i]
                ids2words.append(len(words)-1)
                if i < len(subwords)-1:
                    words.append('') ### word finished prepare new
            else:
                words[-1] += subwords[i]
                ids2words.append(len(words)-1)

            if i==0 or ids2words[i-1]!=ids2words[i]: ### new word
                lids.append([])
                
            lids[-1].append(ids[i])

        assert(len(subwords) == len(ids2words))
        assert(len(words) == len(lids))
        if len(words[-1]) == 0:
            words.pop()
            lids.pop()
        return words, ids2words, subwords, lids
    

if __name__ == '__main__':
    
    token = FlaubertTok()
    
    for l in sys.stdin:
        wrds, lids = token.pretok(l)
        print("{}\t{}".format(' '.join(wrds),lids))
        subwrds, lids = token.subtok(l)
        print("{}\t{}".format(' '.join(subwrds),lids))



        
