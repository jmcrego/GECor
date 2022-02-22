#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import copy
import random
import unicodedata

#from psutil import cpu_count
from Keyboard import Keyboard
from Lexicon import Lexicon
from Vocab import Vocab
from Spacy import Spacy
from Tokenizer import Tokenizer
from collections import defaultdict
from transformers import FlaubertTokenizer
import logging
from Utils import create_logger
import multiprocessing as mp

separ = '￨'
keep = '·'
used = '׃'

class Sentence():
    
    def __init__(self, raw, words, ids, ids2words):
        assert(len(ids) == len(ids2words))
        self.raw = raw     #string   'la maisonette blanche.'
        self.words = words #[string] ['la', 'maisonette',  'blanche', '.']
        #ids               #[int]    [12    34 654 445      223        33]
        #ids2words         #[int]    [0     1  1   1        2          3]
        self.idx2lids = [] #[[int]]  [[12], [34, 654, 445], [223],     [33]]
        if len(ids):
            self.idx2lids.append([ids[0]])
            for i in range(1,len(ids)):
                if ids2words[i] != ids2words[i-1]: ### new word
                    self.idx2lids.append([])
                self.idx2lids[-1].append(ids[i])
        assert(len(words) == len(self.idx2lids))
        self.tags = [keep for _ in self.words]
        self.n_injected = 0

    def word(self, idx):
        return copy.deepcopy(self.words[idx])

    def tag(self, idx):
        return copy.deepcopy(self.tags[idx])

    def lids(self, idx):
        return copy.deepcopy(self.idx2lids[idx])
    
    def __len__(self):
        return len(self.words)

    def allowed(self, idx, vocab=None, min_size=0, isalpha=False, islower=False, isupper=False, hashyphen=False):
        if self.tags[idx] != keep:
            return False
        if min_size > 0 and len(self.words[idx]) < min_size:
            return False
        if vocab is not None and self.words[idx] not in vocab:
            return False
        if isalpha:
            if not self.words[idx].isalpha():
                return False
        if islower:
            if not self.words[idx].islower():
                return False
        if isupper:
            if not self.words[idx].isupper():
                return False
        if hashyphen:
            if not '-' in self.words[idx][1:-1]:
                return False
        return True

    def REPLACE(self, idx, word=None, tag=None, lids=None, count=True):
        if word is not None:
            self.words[idx] = word
        if tag is not None:
            self.tags[idx] = tag
        if lids is not None:
            self.idx2lids[idx] = lids
        if count:
            self.n_injected += 1

    def INJECT(self, idx, word, tag, lids, count=True):
        self.tags.insert(idx, tag)
        self.words.insert(idx, word)
        self.idx2lids.insert(idx, lids)
        if count:
            self.n_injected += 1

    def DELETE(self, idx, count=True):
        self.words.pop(idx)
        self.tags.pop(idx)
        self.idx2lids.pop(idx)
        if count:
            self.n_injected += 1
        
    
class Noiser():

    def __init__(self, lex=None, voc=None, kbd_lc=None, kbd_uc=None, n_noises=None, noises=None, spacy=None, tokenizer=None, accents=None, max_ratio_noise_tokens=0.5, seed=0):
        self.lex = lex
        self.voc = voc
        self.kbd_lc = kbd_lc
        self.kbd_uc = kbd_uc
        self.n_noises = n_noises
        self.noises = noises
        self.spacy = spacy
        self.tokenizer = tokenizer
        self.accents = accents
        self.max_ratio_noise_tokens = max_ratio_noise_tokens
        self.stats_noise_injected = defaultdict(int)
        self.stats_n_injected = defaultdict(int)
        if seed:
            random.seed(seed)

    def __call__(self, raw, toks, ids, ids2toks, nline):

        self.n = nline
        self.sentence = Sentence(raw, toks, ids, ids2toks)
        tok2inf = self.spacy(self.sentence.words) if self.spacy is not None else None            
        inject_n_noises = random.choices([i for i in range(len(self.n_noises))], weights = self.n_noises, k = 1)[0]
        inject_n_noises = min(inject_n_noises, len(self.sentence)*self.max_ratio_noise_tokens)
                
        self.n_attempts = 0
        while self.sentence.n_injected < inject_n_noises and self.n_attempts < inject_n_noises*3:
            
            self.n_attempts += 1
            curr_noise = random.choices(list(self.noises.keys()), weights=list(self.noises.values()), k=1)[0]
            if curr_noise == 'replace:inflection':
                self.do_replace_inflection()
            elif curr_noise == 'replace:homophone':
                self.do_replace_homophone()
            elif curr_noise == 'replace:samepos':
                self.do_replace_samepos()
            elif curr_noise == 'replace:spell':
                self.do_replace_spell()
            elif curr_noise == 'append':
                self.do_append()
            elif curr_noise == 'split':
                self.do_split()
            elif curr_noise == 'inflect':
                self.do_inflect(tok2inf)
            elif curr_noise == 'delete':
                self.do_delete()
            elif curr_noise == 'merge':
                self.do_merge()
            elif curr_noise == 'swap':
                self.do_swap()
            elif curr_noise == 'case:first':
                self.do_case_first()
            elif curr_noise == 'case:upper':
                self.do_case_upper()
            elif curr_noise == 'case:lower':
                self.do_case_lower()
            elif curr_noise == 'hyphen:split':
                self.do_hyphen_split()
            elif curr_noise == 'hyphen:merge':
                self.do_hyphen_merge()
            else:
                sys.stderr.write('error: unparsed noise choice {}'.format(curr_noise))
                sys.exit()

        self.stats_n_injected[self.sentence.n_injected] += 1
        return self.sentence.words, self.sentence.tags, self.sentence.idx2lids


    def stats(self):
        
        sys.stderr.write('N\tNoise_type\n')
        for noise, n in sorted(self.stats_noise_injected.items(), key=lambda x: x[1], reverse=True):
            sys.stderr.write("{}\t{}\n".format(n,noise))
        sys.stderr.write('N_noise\tN\n')
        for k, n in sorted(self.stats_n_injected.items()):
            sys.stderr.write("{}\t{}\n".format(k,n))

    def random_replacement_option(self, replacement_options):
        if len(replacement_options) == 0:
            return '', []
        k = random.randint(0,len(replacement_options)-1)
        new_word = replacement_options[k]
        ids = self.tokenizer.get_ids(new_word)
        if self.tokenizer.is_further_tokenized(ids):
            return '', []
        return new_word, ids

            
    def do_replace_inflection(self):
        ### replace a random word by any in wrd2wrds_same_lemma[word] if word in vocab, tag the previous as REPLACE:INFLECTION_word
        idxs = list(range(len(self.sentence)))
        random.shuffle(idxs)
        for idx in idxs:
            if not self.sentence.allowed(idx, vocab=self.voc):
                continue
            if not self.sentence.allowed(idx, vocab=self.lex.wrd2wrds_same_lemma):
                continue
            old_word = self.sentence.word(idx)
            replacement_options = list(self.lex.wrd2wrds_same_lemma[old_word])
            new_word, ids = self.random_replacement_option(replacement_options)
            if new_word == '':
                continue
            #if len(replacement_options) == 0:
            #    continue            
            #k = random.randint(0,len(replacement_options)-1)
            #new_word = replacement_options[k]
            #ids = self.tokenizer.get_ids(new_word)
            #if self.tokenizer.is_further_tokenized(ids):
            #    continue
            logging.debug('[{}] $REPLACE:INFLECTION {} => {}'.format(self.n,new_word, old_word))
            self.sentence.REPLACE(idx, word=new_word, tag='$REPLACE:INFLECTION_'+old_word, lids=ids)
            self.stats_noise_injected['$REPLACE:INFLECTION'] += 1
            break
        return

    def do_replace_samepos(self): #for PROREL and ART
        ### replace a word any other in wrd2wrds_same_pos[word] if word in vocab and word is pronoun or article, tag the previous as REPLACE:SAMEPOS_word
        idxs = list(range(len(self.sentence)))
        random.shuffle(idxs)
        for idx in idxs:
            if not self.sentence.allowed(idx, vocab=self.voc):
                continue
            if not self.sentence.allowed(idx, vocab=self.lex.wrd2wrds_same_pos):
                continue
            old_word = self.sentence.word(idx)
            replacement_options = list(self.lex.wrd2wrds_same_pos[old_word])
            new_word, ids = self.random_replacement_option(replacement_options)
            if new_word == '':
                continue
            #if len(replacement_options) == 0:
            #    continue
            #k = random.randint(0,len(replacement_options)-1)
            #new_word = replacement_options[k]
            #ids = self.tokenizer.get_ids(new_word)
            #if self.tokenizer.is_further_tokenized(ids):
            #    continue
            logging.debug('[{}] $REPLACE:SAMEPOS {} => {}'.format(self.n,new_word, old_word))
            self.sentence.REPLACE(idx, word=new_word, tag='$REPLACE:SAMEPOS_'+old_word, lids=ids)
            self.stats_noise_injected['$REPLACE:SAMEPOS'] += 1
            break
        return
    
    def do_replace_homophone(self):
        ### replace a random word by any in wrd2wrds_homophones[word] if word in vocab, tag the previous as REPLACE:HOMOPHONE_word
        idxs = list(range(len(self.sentence)))
        random.shuffle(idxs)
        for idx in idxs:
            if not self.sentence.allowed(idx, vocab=self.voc):
                continue
            if not self.sentence.allowed(idx, vocab=self.lex.wrd2wrds_homophones):
                continue
            old_word = self.sentence.word(idx)
            replacement_options = list(self.lex.wrd2wrds_homophones[old_word])
            new_word, ids = self.random_replacement_option(replacement_options)
            if new_word == '':
                continue
            #if len(replacement_options) == 0:
            #    continue
            #k = random.randint(0,len(replacement_options)-1)
            #new_word = replacement_options[k]
            #ids = self.tokenizer.get_ids(new_word)
            #if self.tokenizer.is_further_tokenized(ids):
            #    continue
            logging.debug('[{}] $REPLACE:HOMOPHONE {} => {}'.format(self.n,new_word, old_word))
            self.sentence.REPLACE(idx, word=new_word, tag='$REPLACE:HOMOPHONE_'+old_word, lids=ids)
            self.stats_noise_injected['$REPLACE:HOMOPHONE'] += 1
            break
        return

    def misspell_accents(self, word):
        idxs = list(range(len(word)))
        random.shuffle(idxs)
        for idx in idxs:
            if word[idx] in accents['vowels']:
                letter = word[idx] ### contains the original letter in word
                letter_norm = unicodedata.normalize('NFD', letter)
                letter_norm = letter_norm.encode('ascii', 'ignore')
                letter_norm = letter_norm.decode("utf-8").lower() #no diacritics and lowercase
                if letter_norm not in self.accents:
                    continue
                letters = self.accents[letter_norm]
                while True:
                    k = random.randint(0,len(letters)-1)
                    new_letter = letters[k]
                    if letter.isupper():
                        new_letter = new_letter.upper()
                    if new_letter != letter:
                        word[idx] = new_letter
                        return word
        return word
        
    def misspell(self, word):
        word = list(word)
        r = random.random() ### float between [0, 1)
        if r < 0.5:
            word = self.misspell_accents(word)
            return ''.join(word)
        
        r = random.random() ### float between [0, 1)
        if r < 1.0/4: ### remove character in position k [0, len(word)-1]
            k = random.randint(0,len(word)-1)
            l = word.pop(k)
        elif r < 2.0/4: ### swap characters in positions k and k+1, k in [0, len(word)-2]
            k = random.randint(0,len(word)-2)
            word[k], word[k+1] = word[k+1], word[k] # if word[k] == word[k=1] the returned word is the same!
        elif r < 3.0/4: ### repeat character in position k [0, len(word)-1]
            k = random.randint(0,len(word)-1) 
            word.insert(k,word[k])
        else: ### replace char in positions k [0, len(word)-1] by another close to it
            k = random.randint(0,len(word)-1)
            c = word[k]
            if c in self.kbd_lc:
                near_k, _ = self.kbd_lc.closest(c)
            elif c in self.kbd_uc:
                near_k, _ = self.kbd_uc.closest(c)
            else:
                near_k = self.kbd_lc.all()
            if len(near_k) > 1:
                random.shuffle(near_k)
            word[k] = near_k[0]
        return ''.join(word)
    
    def do_replace_spell(self):
        ### misspell a random word if word in vocab (tag it as REPLACE:SPELL_word)
        idxs = list(range(len(self.sentence)))
        random.shuffle(idxs)
        for idx in idxs:
            if not self.sentence.allowed(idx, vocab=self.voc, min_size=2):
                continue
            old_word = self.sentence.word(idx)
            new_word = self.misspell(old_word)
            if new_word == old_word:
                continue
            ids = self.tokenizer.get_ids(new_word)
            if self.tokenizer.is_further_tokenized(ids):
                continue
            #print(idx, new_word, ids)
            logging.debug('[{}] $REPLACE:SPELL {} => {}'.format(self.n,new_word, old_word))
            self.sentence.REPLACE(idx, word=new_word, tag='$REPLACE:SPELL_'+old_word,lids=ids)
            self.stats_noise_injected['$REPLACE:SPELL'] += 1
            break
        return

    def do_append(self):
        ### remove the next word if word in vocab, tag the previous as APPEND_word
        idxs = list(range(len(self.sentence)-1))
        random.shuffle(idxs)
        for idx in idxs:
            if not self.sentence.allowed(idx) or not self.sentence.allowed(idx+1, vocab=self.voc) or not self.sentence.allowed(idx+1, vocab=self.lex.wrd2pos):
                continue
            deleted_word = self.sentence.word(idx+1)
            pos = self.lex.wrd2pos[deleted_word] ###is a set
            if 'ADJ' in pos or 'NOM' in pos or 'VERB' in pos or 'ADV' in pos or 'AUX' in pos or 'ONO' in pos: ### do not append VERB NOM ADV ADJ AUX ONO (tags in lexicon)
                continue
            new_tag = '$APPEND_'+deleted_word
            logging.debug('[{}] $APPEND {}'.format(self.n,deleted_word))
            self.sentence.REPLACE(idx, tag=new_tag)
            self.sentence.DELETE(idx+1, count=False)
            self.stats_noise_injected['$APPEND'] += 1
            break
        return

    def do_split(self):
        ### join two consecutive vocab (w1 w2) if w1 and w2 in vocab, tag as SPLIT_w1
        idxs = list(range(len(self.sentence)-1))
        random.shuffle(idxs)
        for idx in idxs:
            if not self.sentence.allowed(idx, vocab=self.voc) or not self.sentence.allowed(idx+1, vocab=self.voc):
                continue
            new_word =  self.sentence.word(idx) + self.sentence.word(idx+1)
            ids = self.tokenizer.get_ids(new_word)
            if self.tokenizer.is_further_tokenized(ids):
                continue
            #print(idx, new_word, ids)
            logging.debug('[{}] $SPLIT {} => {} {}'.format(self.n,new_word,self.sentence.word(idx),self.sentence.word(idx+1)))
            new_tag = '$SPLIT_' + self.sentence.word(idx)
            self.sentence.REPLACE(idx, word=new_word, tag=new_tag, lids=ids)
            self.sentence.DELETE(idx+1, count=False)            
            self.stats_noise_injected['$SPLIT'] += 1
            break
        return 0

    def reinflect(self, txt, inflection):
        infl = inflection.split(';') #exemple;NOUN;Gender=Masc;Number=Sing (Spacy tags)
        lem = infl[0]
        pos = infl[1]
        for wrd in self.lex.lempos2wrds[lem+self.lex.separ+pos]:
            if wrd != txt:
                return wrd
        return ''

    def do_inflect(self,tok2inf):
        if tok2inf is None:
            return
        ### replace word i if in vocab by another with different inflection, tag it with INFLECT:inflection
        idxs = list(range(len(self.sentence)))
        random.shuffle(idxs)
        for idx in idxs:
            if not self.sentence.allowed(idx, vocab=tok2inf):
                continue
            old_word = self.sentence.word(idx)
            curr_inflection = tok2inf[old_word]
            new_word = self.reinflect(old_word, curr_inflection)
            if new_word == '':
                continue
            ids = self.tokenizer.get_ids(new_word)
            if self.tokenizer.is_further_tokenized(ids):
                continue
            #print(idx, new_word, ids)
            curr_inflection = ';'.join(curr_inflection.split(';')[1:]) ### discard lemma
            logging.debug('[{}] $INFLECT:{} {} => {}'.format(self.n,curr_inflection,new_word, old_word))
            new_tag = '$INFLECT:' + curr_inflection
            self.sentence.REPLACE(idx, word=new_word, tag=new_tag)
            self.stats_noise_injected[new_tag] += 1
            break
        return

    def do_delete(self):
        ### insert a random word (tag it as DELETE)
        idxs = list(range(len(self.sentence)))
        random.shuffle(idxs)
        for idx in idxs:
            if not self.sentence.allowed(idx):
                continue
            if random.random() < 0.5: ### random word                
                lvocab = list(self.voc.vocab)
                new_word = lvocab[random.randint(0,len(lvocab)-1)]
            else: ### copy word
                new_word = self.sentence.word(idx)
            ids = self.tokenizer.get_ids(new_word)
            if self.tokenizer.is_further_tokenized(ids):
                continue
            logging.debug('[{}] $DELETE {}'.format(self.n,new_word))
            self.sentence.INJECT(idx, word=new_word, tag='$DELETE', lids=ids)
            self.sentence.REPLACE(idx+1, tag=used, lids=ids)
            self.stats_noise_injected['$DELETE'] += 1
            break
        return

    def do_merge(self):
        ### split toks[idx] in two tokens and tag the first to MERGE
        idxs = list(range(len(self.sentence)))
        random.shuffle(idxs)
        for idx in idxs:
            if not self.sentence.allowed(idx, vocab=self.voc, min_size=2):
                continue
            curr_word = self.sentence.word(idx)
            k = random.randint(1,len(curr_word)-1)
            ls = curr_word[:k]
            ls_ids = self.tokenizer.get_ids(ls)
            if self.tokenizer.is_further_tokenized(ls_ids):
                continue
            #print(idx, ls, ls_ids)
            rs = curr_word[k:]
            rs_ids = self.tokenizer.get_ids(rs)
            if self.tokenizer.is_further_tokenized(rs_ids):
                continue
            #print(idx, rs, rs_ids)
            logging.debug('[{}] $MERGE {} {} => {}'.format(self.n,ls, rs, curr_word))
            self.sentence.REPLACE(idx, word=''.join(rs), lids=rs_ids, tag=used)
            self.sentence.INJECT(idx, word=''.join(ls), tag='$MERGE', lids=ls_ids, count=False)
            self.stats_noise_injected['$MERGE'] += 1
            break
        return

    def do_swap(self):
        ### swap tokens i and i+1 (tag the first as SWAP)
        idxs = list(range(len(self.sentence)-1))
        random.shuffle(idxs)
        for idx in idxs:
            if not self.sentence.allowed(idx) or not self.sentence.allowed(idx+1):
                continue
            curr_word = self.sentence.word(idx)
            curr_tag = self.sentence.tag(idx)
            curr_ids = self.sentence.lids(idx)
            next_word = self.sentence.word(idx+1)
            next_ids = self.sentence.lids(idx+1)
            if curr_word == next_word:
                continue
            logging.debug('[{}] $SWAP {} {} => {} {}'.format(self.n,next_word, curr_word, curr_word, next_word))
            self.sentence.REPLACE(idx, word=next_word, tag='$SWAP', lids=next_ids)
            self.sentence.REPLACE(idx+1, word=curr_word, tag=used, lids=curr_ids, count=False)
            self.stats_noise_injected['$SWAP'] += 1
            break
        return
    
    def do_case_first(self):
        ### change the case of first char in token
        idxs = list(range(len(self.sentence)))
        random.shuffle(idxs)
        for idx in idxs:
            if not self.sentence.allowed(idx,min_size=2,vocab=self.voc,isalpha=True):
                continue
            curr_word = self.sentence.word(idx)
            first = curr_word[0]
            rest = curr_word[1:]
            first = first.lower() if first.isupper() else first.upper()
            ids = self.tokenizer.get_ids(rest+first)
            if self.tokenizer.is_further_tokenized(ids):
                continue
            #print(idx, first+rest, ids)
            logging.debug('[{}] $CASE:FIRST {} => {}'.format(self.n,first+rest, curr_word))
            self.sentence.REPLACE(idx,word=first+rest, lids=ids, tag='$CASE:FIRST')
            self.stats_noise_injected['$CASE:FIRST'] += 1
            break
        return

    def do_case_upper(self):
        ### lower case all chars in a (uppercased) token and tag it as CASEUPPER
        idxs = list(range(len(self.sentence)))
        random.shuffle(idxs)
        for idx in idxs:
            if not self.sentence.allowed(idx,vocab=self.voc,isalpha=True,isupper=True):
                continue
            curr_word =	self.sentence.word(idx).lower()
            ids = self.tokenizer.get_ids(curr_word)
            if self.tokenizer.is_further_tokenized(ids):
                continue
            #print(idx, curr_word, ids)
            logging.debug('[{}] $CASE:UPPER {} => {}'.format(self.n,curr_word, curr_word.upper()))
            self.sentence.REPLACE(idx,word=curr_word, lids=ids, tag='$CASE:UPPER')
            self.stats_noise_injected['$CASE:UPPER'] += 1
            break
        return

    def do_case_lower(self):
        ### upper case all chars in a (lowercased) token and tag it as CASELOWER
        idxs = list(range(len(self.sentence)))
        random.shuffle(idxs)
        for idx in idxs:
            if not self.sentence.allowed(idx,vocab=self.voc,isalpha=True,islower=True):
                continue
            curr_word =	self.sentence.word(idx).upper()
            ids = self.tokenizer.get_ids(curr_word)
            if self.tokenizer.is_further_tokenized(ids):
                continue
            #print(idx, curr_word, ids)
            logging.debug('[{}] $CASE:LOWER {} => {}'.format(self.n,curr_word, curr_word.lower()))
            self.sentence.REPLACE(idx,word=curr_word, lids=ids, tag='$CASE:LOWER')
            self.stats_noise_injected['$CASE:LOWER'] += 1
            break
        return

    def do_hyphen_split(self):
        ### take two consecutive words and join them with an hyphen, tag it as HYPHEN_SPLIT
        idxs = list(range(len(self.sentence)-1))
        random.shuffle(idxs)
        for idx in idxs:
            if not self.sentence.allowed(idx,vocab=self.voc,isalpha=True) or not self.sentence.allowed(idx+1,vocab=self.voc,isalpha=True):
                continue
            curr_word = self.sentence.word(idx)
            next_word = self.sentence.word(idx+1)
            ids = self.tokenizer.get_ids(curr_word+'-'+next_word)
            if self.tokenizer.is_further_tokenized(ids):
                continue
            #print(idx, curr_word+'-'+next_word, ids)
            logging.debug('[{}] $HYPHEN:SPLIT {} => {} {}'.format(self.n,curr_word+'-'+next_word, curr_word, next_word))
            self.sentence.REPLACE(idx, word=curr_word+'-'+next_word, tag='$HYPHEN:SPLIT', lids=ids)
            self.sentence.DELETE(idx+1, count=False)            
            self.stats_noise_injected['$HYPHEN:SPLIT'] += 1
            break
        return

    def do_hyphen_merge(self):
        ### take a word with an hyphen and split it into two, tag the first as HYPHEN_MERGE
        idxs = list(range(len(self.sentence)))
        random.shuffle(idxs)
        for idx in idxs:
            if not self.sentence.allowed(idx,hashyphen=True):
                continue
            curr_word = self.sentence.word(idx)
            
            p = curr_word[1:-1].find('-') + 1
            if p == -1:
                continue
            first = curr_word[:p]
            ids_first = self.tokenizer.get_ids(first)
            if self.tokenizer.is_further_tokenized(ids_first):
                continue
            second = curr_word[p+1:]
            ids_second = self.tokenizer.get_ids(second)
            if self.tokenizer.is_further_tokenized(ids_second):
                continue
            if not first in self.voc or not second in self.voc:
                continue
            logging.debug('[{}] $HYPHEN:MERGE {} {} => {}'.format(self.n,first, second, curr_word))
            self.sentence.REPLACE(idx, word=second, lids=ids_second, tag=used)
            self.sentence.REPLACE(idx, word=first, lids=ids_first, tag='$HYPHEN:MERGE', count=False)
            self.stats_noise_injected['$HYPHEN:MERGE'] += 1
            break
        return




if __name__ == '__main__':

    create_logger(None,'info')
    seed = 128
    accents = {'a':'aàáâä', 'e':'eéèêë', 'i':'iíìîï', 'o':'oóòôö', 'u':'uúùûü'}
    vowels = sum([[c for c in l] for l in list(accents.values())], []) ### flattens list of chars
    accents['vowels'] = vowels
    kbd_lc = Keyboard(['`1234567890-=', 'qwertyuiop[]\\', 'asdfghjkl;\'', 'zxcvbnm,./'], [0, 1.5, 1.85, 2.15], 2.0)
    kbd_uc = Keyboard(['~!@#$%^&*()_+', 'QWERTYUIOP{}|',  'ASDFGHJKL:"',  'ZXCVBNM<>?'], [0, 1.5, 1.85, 2.15], 2.0)
    lex = Lexicon('resources/Lexique383.tsv') #Lexique383.tsv Morphalou3.1_CSV.csv
    voc = Vocab('resources/french.dic.50k')
    spacy = Spacy(model="fr_core_news_md", pos_to_consider=['NOUN','ADJ','VERB','AUX'])
    n_noises = [1,5,5,5,5,5]
    max_ratio_noise_tokens = 0.5
    noises = {'replace:inflection':10, 'replace:homophone':10, 'replace:samepos':5, 'replace:spell':10, 'append':5, 'split':5, 'inflect':10, 'delete':5, 'merge':5, 'swap':5, 'case:first':3, 'case:upper':3, 'case:lower':3, 'hyphen:split':3, 'hyphen:merge':3}
    tokenizer = Tokenizer('/gpfsdswork/projects/rech/sfz/utt84zy/GEC/GEC/tokenizers')
    
    ### Multi_process_setup
    cpu_count_ = mp.cpu_count() # number of available cpus
    print("%d cpus are available"%cpu_count_)
    file_name = sys.argv[1]
    file_size = os.path.getsize(file_name)
    chunk_size = file_size // cpu_count_
    chunk_args = []
    
    ### Apply noise to a line
    def process_line(noiser, line):
        #line = line.encode('utf8').decode('utf8')
        l = line.rstrip()
        ids = tokenizer.get_ids(l)
        words, ids2words, _ = tokenizer.get_words_ids2words_subwords(ids)
        noisy_words, noisy_tags, noisy_idx2lids = noiser(l, words, ids, ids2words, 1)
        noisy_wordstags = [noisy_words[i]+separ+(noisy_tags[i].replace(used,keep)) for i in range(len(noisy_words))]
        print(('{}\t{}'.format(' '.join(noisy_wordstags), noisy_idx2lids)), flush=True)
        
        
    ### Apply noise to a chunk of the original file and return the statistic of noise used in that chunk
    def process_chunk(file_name, chunk_start, chunk_end):
        noiser = Noiser(lex=lex, voc=voc, kbd_lc=kbd_lc, kbd_uc=kbd_uc, n_noises=n_noises, noises=noises, spacy=spacy, tokenizer=tokenizer, max_ratio_noise_tokens=max_ratio_noise_tokens, accents=accents, seed=seed)
        
        with open(file_name, 'r') as f:
            # Moving stream position to `chunk_start`
            f.seek(chunk_start)

            # Read and process lines until `chunk_end`
            for line in f:
                chunk_start += len(line)
                if chunk_start > chunk_end:
                    break
                process_line(noiser, line)
        return noiser.stats_noise_injected, noiser.stats_n_injected

    ### Split original file to many chunks of chunk_size
    with open(file_name, 'r', encoding="latin1") as f:
        def is_start_of_line(position):
            if position==0:
                return True
            f.seek(position-1)
            return f.read(1)=='\n'
        
        def get_next_line_position(position):
            f.seek(position)
            f.readline()
            return f.tell()
        
        chunk_start = 0
        sent_id = 0
        while chunk_start<file_size:
            chunk_end = min(file_size,chunk_start+chunk_size)

            while not is_start_of_line(chunk_end):
                chunk_end -=1
            
            # Handle the case when a line is too long to fit the chunk size
            if chunk_start == chunk_end:
                chunk_end = get_next_line_position()
    
            # Save `process_chunk` arguments
            args = (file_name, chunk_start, chunk_end)
            chunk_args.append(args)

            chunk_start = chunk_end
    
    ### Set up and launch noiser process on each cpu
    with mp.Pool(cpu_count_) as p:
        # Run chunks in parallel
        chunk_results = p.starmap(process_chunk, chunk_args)

    # Combine chunks' statistics into `results`
    stats_noise_injected = {}
    stats_n_injected = {}
    
    for chunk_result in chunk_results:
      chunk_stats_noise_injected, chunk_stats_n_injected = chunk_result
      for item in chunk_stats_noise_injected.items():
        stats_noise_injected[item[0]] = stats_noise_injected.get(item[0],0) + item[1]
      for item in chunk_stats_n_injected.items():
        stats_n_injected[item[0]] = stats_n_injected.get(item[0],0) + item[1]
    
    sys.stderr.write('N\tNoise_type\n')
    for noise, n in sorted(stats_noise_injected.items(), key=lambda x: x[1], reverse=True):
        sys.stderr.write("{}\t{}\n".format(n,noise))
    sys.stderr.write('N_noise\tN\n')
    for k, n in sorted(stats_n_injected.items()):
        sys.stderr.write("{}\t{}\n".format(k,n))
