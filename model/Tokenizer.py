import sys
from transformers import FlaubertTokenizer


class Tokenizer():

    def __init__(self, modelname, do_lowercase=False):
        self.flaubert_tokenizer = FlaubertTokenizer.from_pretrained(modelname, do_lowercase=do_lowercase)
        
    def get_str(self, ids, remove_eow=True):
        t = ''.join(self.flaubert_tokenizer.convert_ids_to_tokens(ids))
        if remove_eow:
            t = t.replace('</w>','')
        return t

    def get_pretok(self, ids):
        return ''.join(self.flaubert_tokenizer.convert_ids_to_tokens(ids)).replace('</w>',' ')
    
    def get_ids(self, l, is_split_into_words=False, add_special_tokens=False):
        #l : string
        '''
        Returns:
        ids [int] : list of ints
        '''
        return self.flaubert_tokenizer(l,is_split_into_words=is_split_into_words, add_special_tokens=add_special_tokens)['input_ids']

    def is_further_tokenized(self, ids):
        subwords = self.flaubert_tokenizer.convert_ids_to_tokens(ids)
        #print(' '.join(subwords))
        for i in range(len(subwords)-1):
            if subwords[i].endswith('</w>'):
                return True
        return False
        
    def get_words_ids2words_subwords(self, ids):
        #ids [int] : list of ints
        '''
        Returns:
        words     [string] : list of strings
        ids2words [int] : list of ints
        subwords  [string] : list of strings
        '''
        subwords = self.flaubert_tokenizer.convert_ids_to_tokens(ids)
        assert(len(ids) == len(subwords))
        ids2words = []
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
                
        assert(len(subwords) == len(ids2words))
        if len(words[-1]) == 0:
            words.pop()
        return words, ids2words, subwords

if __name__ == '__main__':

    t = Tokenizer('flaubert/flaubert_base_cased')
    
    for l in sys.stdin:
        ids = t.get_ids(l.rstrip())
        print('ids',ids)
        words, ids2words, subwords = t.get_words_ids2words_subwords(ids)
        print('words',words)
        print('ids2words',ids2words)
        print('subwords',subwords)

    #print('Word')
    #word = 'anticonstitutionnellement'
    #ids = t.get_ids(word)
    #print('ids',ids)
    #words, ids2words, subwords = t.get_words_ids2words_subwords(ids)
    #print('words',words)
    #print('ids2words',ids2words)
    #print('subwords',subwords)

