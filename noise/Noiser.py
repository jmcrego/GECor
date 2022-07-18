import sys
import json
import time
import spacy
import random
import logging
#import argparse
from collections import defaultdict
from utils.FlaubertTok import FlaubertTok
from utils.Lexicon import Lexicon
from utils.Utils import MAX_IDS_LEN, KEEP, shape_of_word, reshape, PUNCTUATION
from noise.LemRules import LemRules
from noise.Misspell import Misspell
from noise.Homophones import Homophones
from model.Vocab import Vocab


class Noiser():
    def __init__(self, noises, lem, lex, pho, voce, vocc, vocl, vocs, max_noises=0, max_ratio=0.5, p_cor=0.0, p_lin=0.0, seed=0):
        if seed:
            random.seed(seed)
        self.flauberttok = FlaubertTok(max_ids_len=MAX_IDS_LEN)
        self.lexicon = Lexicon(lex)
        self.lemrules = LemRules(lem,self.lexicon) if lem is not None else None
        self.homophones = Homophones(pho)
        self.vocab_err = Vocab(voce) if voce is not None else None
        self.vocab_cor = Vocab(vocc) if vocc is not None else None
        self.vocab_lin = Vocab(vocl) if vocl is not None else None
        self.vocab_sha = Vocab(vocs) if vocs is not None else None
        self.misspell = Misspell()
        self.noises = [x[0] for x in noises]
        self.wnoises = [x[1] for x in noises]
        self.max_noises = max_noises
        self.max_ratio = max_ratio
        self.p_cor = p_cor
        self.p_lin = p_lin
        self.n_sents = 0
        self.n_tokens = 0
        self.n_tokens_noised_with = defaultdict(int) #n_tokens_noised_with['$APND'] += 1
        self.n_sents_with_n_noises = defaultdict(int) #n_sents_with_n_noises[3] += 1
        logging.info("Built Noiser")

    def __call__(self, ldict):
        self.s = ldict
        self.n_sents += 1
        self.n_tokens += len(self.s)
        n_noises_sentence = 0
        n_noises_toinject = random.randint(0, min(self.max_noises, int(len(self.s)*self.max_ratio)))
        logging.debug("BEFORE: {}".format([d['r'] for d in ldict]))
        for _ in range(n_noises_toinject*2): ### try this maximum number of times before stopping
            if n_noises_sentence >= n_noises_toinject:
                break
            idxs = [idx for idx in range(len(self.s)) if 'E' not in self.s[idx]]
            if len(idxs) == 0:
                break
            random.shuffle(idxs)
            idx = idxs[0]
            noise = random.choices(self.noises,self.wnoises)[0] #select the noise to inject
            if noise == 'phon': ### replace current word by homophone and add $PHON
                n_noises_sentence +=self.inject_phon(idx)
            elif noise == 'lemm': ### replace current word by another with same lemma and add $LEMM
                n_noises_sentence += self.inject_lemm(idx)
            elif noise == 'apnd': ### dele next word and add $APND error to current word (correction is deled word)
                n_noises_sentence += self.inject_apnd(idx)
            elif noise == 'spel': ### misspel current word and add $SPEL (correction is original word)
                n_noises_sentence += self.inject_spel(idx)
            elif noise == 'splt': ### mrge current and next word into one tokens and add $SPLT error to resulting word (correction is first mrged word)
                n_noises_sentence += self.inject_splt(idx)
            elif noise == 'dele': ### copy word and mark the first as $DELE
                n_noises_sentence += self.inject_dele(idx)
            elif noise == 'mrge': ### splt current word into two tokens and add $MRGE error to first splt
                n_noises_sentence += self.inject_mrge(idx)
            elif noise == 'hyph': ### splt/mrge words using hyph and add $HYPHs/$HYPHm
                n_noises_sentence += self.inject_hyph(idx)
            elif noise == 'swap': ### swap current/next words and add $SWAP error to first
                n_noises_sentence += self.inject_swap(idx)
            elif noise == 'case': ### change case of current word and add $CASE1/$CASEn
                n_noises_sentence += self.inject_case(idx)
        self.n_sents_with_n_noises[n_noises_sentence] += 1
        logging.debug("AFTER : {}".format([d['r'] for d in ldict]))
        #remove all p and plm from words
#        for i in range(len(self.s)):
#            if 'plm' in self.s[i]: #not used anymore
#                self.s[i].pop('plm')
#            if 'E' not in self.s[i]:
#                self.s[i]['E'], self.s[i]['iE'] = self.buildE(KEEP)
                
        return self.s

    def debug(self,tic):
        toc = time.time()
        n_tokens_noised = sum(v for k,v in self.n_tokens_noised_with.items())
        n_sents_noised = sum(v for k,v in self.n_sents_with_n_noises.items()) - self.n_sents_with_n_noises[0]
        
        logging.info('Parsed {}/{} snts/toks in {:.2f} sec ({:.2f} snts/sec {:.2f} toks/sec)'.format(self.n_sents,self.n_tokens,toc-tic,self.n_sents/(toc-tic),self.n_tokens/(toc-tic)))
        if n_tokens_noised == 0:
            logging.info('No noises injected')
            return
        
        logging.info('Noised {} snts ({:.2f}%)'.format(n_sents_noised,100.0*n_sents_noised/self.n_sents))
        for n, N in sorted(self.n_sents_with_n_noises.items()):
            logging.info('\t{}-noises\t{} snts ({:.2f}%)'.format(n,N,100.0*N/self.n_sents))

        logging.info('Noised {} toks ({:.2f}%)'.format(n_tokens_noised,100.0*n_tokens_noised/self.n_tokens,100.0*n_tokens_noised/self.n_tokens))
        for noise,n in self.n_tokens_noised_with.items():
            logging.info('\t{}   \t{} toks ({:.2f}%)'.format(noise,n,100.0*n/n_tokens_noised))


    def buildE(self, err):
        E = str(err)
        iE = int(self.vocab_err[err]) if self.vocab_err is not None else None
        return E, iE

    def buildS(self, idx):
        if isinstance(idx,int):
            s = str(self.s[idx]['s'])
        else: ###idx is the string to compute the shape on
            s = shape_of_word(idx)
        i_s = int(self.vocab_sha[s]) if self.vocab_sha is not None else None
        return s, i_s
    
    def buildC(self, idx):
        C = str(self.s[idx]['r'])
        iC = int(self.vocab_cor[self.s[idx]['r']]) if self.vocab_cor is not None else None
        iCC = list(self.s[idx]['i'])
        return C, iC, iCC
        
    def buildL(self, idx):
        if 'plm' not in self.s[idx]:
            return None, None
        l = self.s[idx]['plm'].split('|')
        l.pop(1)
        L = str('|'.join(l))
        iL = int(self.vocab_lin[L]) if self.vocab_lin is not None else None
        return L, iL
            
    def dword_create(self, r, s, i, t, i_s=None, E=None, iE=None, C=None, iC=None, iCC=None, L=None, iL=None, p_cor=1.0, p_lin=1.0):
        d = {'r':str(r), 's':str(s), 'i':list(i), 't':str(t)}
        if E is not None:
            d['E'] = str(E)
            if iE is not None:
                d['iE'] = int(iE)
        if i_s is not None:
            d['is'] = int(i_s)
        if random.random() < p_cor:
            if C is not None:
                d['C'] = str(C)
                if iC is not None:
                    d['iC'] = int(iC)
                if iCC is not None:
                    d['iCC'] = list(iCC)
        if random.random() < p_lin:
            if L is not None:
                d['L'] = str(L)
                if iL is not None:
                    d['iL'] = int(iL)
        return d

    #########################################
    ### $APND ############################### r, s, i, t, E, C, iC, (plm, p)
    #########################################
    def inject_apnd(self, idx):
        if idx >= len(self.s):
            return False
        if 'E' in self.s[idx]:
            return False
        if idx+1 >= len(self.s):
            return False
        if 'E' in self.s[idx+1]:
            return False
        if 'plm' not in self.s[idx+1] and self.s[idx+1]['r'] not in PUNCTUATION: #i don't append words without linguistic annotations unless punctuation
            return False
        if 'plm' in self.s[idx+1] and self.s[idx+1]['plm'].split('|')[0] in ['VER', 'ADJ', 'NOM']: #i don't append open class words
            return False
        if self.vocab_cor is not None and self.s[idx+1]['r'] not in self.vocab_cor: #i don't append words that i cannot generate
            return False
        #
        r = str(self.s[idx]['r'])
        i = list(self.s[idx]['i'])
        t = str(self.s[idx]['t'])
        #E = '$APND'
        E, iE = self.buildE('$APND')
        s, i_s = self.buildS(idx)
        C, iC, iCC = self.buildC(idx+1)
        L, iL = self.buildL(idx)
        self.s[idx] = self.dword_create(r, s, i, t, i_s=i_s, E=E, iE=iE, C=C, iC=iC, iCC=iCC, L=L, iL=iL, p_cor=1.0, p_lin=self.p_lin)
        self.s.pop(idx+1) #delete idx+1
        #
        self.n_tokens_noised_with[E] += 1
        logging.debug('{}\t{}'.format(E,C))
        return True
        
    #########################################
    ### $DELE ############################### r, s, i, t, E, C, iC, (plm, p)
    #########################################
    def inject_dele(self, idx):
        if idx >= len(self.s):
            return False
        if 'E' in self.s[idx]:
            return False
        #
        r = str(self.s[idx]['r'])
        i = list(self.s[idx]['i'])
        t = str(self.s[idx]['t'])
        #E = '$DELE'
        E, iE = self.buildE('$DELE')
        E2, iE2 = self.buildE(KEEP)
        s, i_s = self.buildS(idx)
        C, iC, iCC = self.buildC(idx)
        L, iL = self.buildL(idx)
        self.s.pop(idx)
        self.s.insert(idx, self.dword_create(r, s, i, t, i_s=i_s, E=E, iE=iE, C=C, iC=iC, iCC=iCC, L=L, iL=iL, p_cor=self.p_cor, p_lin=self.p_lin))
        self.s.insert(idx, self.dword_create(r, s, i, t, i_s=i_s, E=E2, iE=iE2, C=C, iC=iC, iCC=iCC, L=L, iL=iL, p_cor=self.p_cor, p_lin=self.p_lin))
        #
        self.n_tokens_noised_with[E] += 1
        logging.debug('{}\t{}'.format(E,C))
        return True

    #########################################
    ### $SWAP ############################### r, s, i, t, E, C, iC, (plm, p)
    #########################################
    def inject_swap(self, idx):
        if idx >= len(self.s):
            return False
        if 'E' in self.s[idx]:
            return False
        if idx+1 >= len(self.s):
            return False
        if 'E' in self.s[idx+1]:
            return False
        #
        ra = str(self.s[idx]['r'])
        ia = list(self.s[idx]['i'])
        ta = str(self.s[idx]['t']) if 't' in self.s[idx] else None
        #Ea = KEEP
        Ea, iEa = self.buildE(KEEP)
        sa, i_sa = self.buildS(idx)
        Ca, iCa, iCCa = self.buildC(idx)
        La, iLa = self.buildL(idx)
        #
        rb = str(self.s[idx+1]['r'])
        ib = list(self.s[idx+1]['i'])
        tb = str(self.s[idx+1]['t']) if 't' in self.s[idx+1] else None
        #Eb = '$SWAP'
        Eb, iEb = self.buildE('$SWAP')
        sb, i_sb = self.buildS(idx+1)
        Cb, iCb, iCCb = self.buildC(idx+1)
        Lb, iLb = self.buildL(idx+1)
        #
        self.s.pop(idx) #deletes a
        self.s.pop(idx) #deletes b
        self.s.insert(idx, self.dword_create(ra, sa, ia, t=ta, i_s=i_sa, E=Ea, iE=iEa, C=Ca, iC=iCa, iCC=iCCa, L=La, iL=iLa, p_cor=self.p_cor, p_lin=self.p_lin)) #inserts a
        self.s.insert(idx, self.dword_create(rb, sb, ib, t=tb, i_s=i_sb, E=Eb, iE=iEb, C=Cb, iC=iCb, iCC=iCCb, L=Lb, iL=iLb, p_cor=self.p_cor, p_lin=self.p_lin)) #inserts b a
        #
        self.n_tokens_noised_with[Eb] += 1
        logging.debug('{}\t{} <-> {}'.format(Eb,rb,ra))
        return True

    #########################################
    ### $LEMM ############################### r, s, i, t, E, C, iC, (plm, p)
    #########################################
    def inject_lemm(self, idx):
        if idx >= len(self.s):
            return False
        if 'E' in self.s[idx]:
            return False
        if 'plm' not in self.s[idx]:
            return False
        if self.lemrules is None:
            return False
        if self.vocab_cor is not None and self.s[idx]['r'] not in self.vocab_cor: #i don't append words that i cannot generate
            return False
        t, new_plm = self.lemrules(self.s[idx])
        if t is None:
            return False
        old_plm = str(self.s[idx]['plm'])
        #
        r = reshape(t,self.s[idx]['s'])
        i = self.flauberttok.ids(r, is_split_into_words=True)
        #E = '$LEMM'
        E, iE = self.buildE('$LEMM')
        s, i_s = self.buildS(r)
        C, iC, iCC = self.buildC(idx)
        L, iL = self.buildL(idx)        
        logging.debug("L={} iL={}".format(L,iL))
        #
        self.s[idx] = self.dword_create(r, s, i, t, i_s=i_s, E=E, iE=iE, C=C, iC=iC, iCC=iCC, L=L, iL=iL, p_cor=1.0, p_lin=self.p_lin)
        #
        self.n_tokens_noised_with[E] += 1
        logging.debug('{}\t{} -> {}\t{} -> {}'.format(E,C,r,old_plm,new_plm))
        return True

    #########################################
    ### $SPEL ############################### r, s, i, t, E, C, iC, (plm, p)
    #########################################
    def inject_spel(self, idx):
        if idx >= len(self.s):
            return False
        if 'E' in self.s[idx]:
            return False
        if self.vocab_cor is not None and self.s[idx]['r'] not in self.vocab_cor: #i don't append words that i cannot generate
            return False
        if not self.s[idx]['r'].isalpha() and self.s[idx]['r'] not in PUNCTUATION: ### alphabetic or punctuation (attention with hyphens)
            return False
        r = self.misspell(self.s[idx]['r'])
        if r is None:
            return False
        #
        i = self.flauberttok.ids(r,is_split_into_words=True)
        t = self.lexicon.inlexicon(r)
        #E = '$SPEL'
        E, iE = self.buildE('$SPEL')
        s, i_s = self.buildS(r)
        C, iC, iCC = self.buildC(idx)
        L, iL = self.buildL(idx)        
        #
        self.s[idx] = self.dword_create(r, s, i, t, i_s=i_s, E=E, iE=iE, C=C, iC=iC, iCC=iCC, L=L, iL=iL, p_cor=1.0, p_lin=self.p_lin)
        #
        self.n_tokens_noised_with[E] += 1
        logging.debug('{}\t{} -> {}'.format(E,C,r))
        return True

    #########################################
    ### $PHON ############################### r, s, i, t, E, C, iC, (plm, p)
    #########################################
    def inject_phon(self, idx):
        if idx >= len(self.s):
            return False
        if 'E' in self.s[idx]:
            return False
        if self.vocab_cor is not None and self.s[idx]['r'] not in self.vocab_cor: #i don't append words that i cannot generate
            return False
        if self.s[idx] == '': #not in lexicon
            return False
        t = self.homophones(self.s[idx]['t'])
        if t is None:
            return False
        #
        r = reshape(t,self.s[idx]['s'])
        i = self.flauberttok.ids(r, is_split_into_words=True)
        #E = '$PHON'
        E, iE = self.buildE('$PHON')
        s, i_s = self.buildS(r)
        C, iC, iCC = self.buildC(idx)
        L, iL = self.buildL(idx)        
        #
        self.s[idx] = self.dword_create(r, s, i, t, i_s=i_s, E=E, iE=iE, C=C, iC=iC, iCC=iCC, L=L, iL=iL, p_cor=1.0, p_lin=self.p_lin)
        #
        self.n_tokens_noised_with[E] += 1
        logging.debug('{}\t{} -> {}'.format(E,C,r))
        return True
        
    #########################################
    ### $CASE ############################### r, s, i, t, E, C, iC, (plm, p)
    #########################################
    def inject_case(self, idx):
        if idx >= len(self.s):
            return False
        if 'E' in self.s[idx]:
            return False
        if self.vocab_cor is not None and self.s[idx]['r'] not in self.vocab_cor: #i don't append words that i cannot generate
            return False
        #
        old_wrd = str(self.s[idx]['r'])
        old_shape = str(self.s[idx]['s'])
        if len(old_shape) == 1: ### all lowercased or uppercased
            if len(old_wrd) == 1:
                #E = '$CASEn'
                E, iE = self.buildE('$CASEn')
                if old_shape == 'X':
                    r = old_wrd.lower()
                elif old_shape == 'x':
                    r = old_wrd.upper()
                else:
                    return False
            else: #len(old_wrd)>1
                if random.random() < 0.5: #CASEn
                    #E = '$CASEn'
                    E, iE = self.buildE('$CASEn')
                    if old_shape == 'X':
                        r = old_wrd.lower()
                    elif old_shape == 'x':
                        r = old_wrd.upper()
                    else:
                        return False
                else: #CASE1
                    #E = '$CASE1'
                    E, iE = self.buildE('$CASE1')
                    if old_shape == 'X':
                        r = old_wrd[0].lower() + old_wrd[1:]
                    elif old_shape == 'x':
                        r = old_wrd[0].upper() + old_wrd[1:]
                    else:
                        return False
        else: #len(txt_shape) > 1 (i can only do CASE1)
            #E = '$CASE1'
            E, iE = self.buildE('$CASE1')
            if old_shape.startswith('X'):
                r = old_wrd[0].lower() + old_wrd[1:]
            elif old_shape.startswith('x'):
                r = old_wrd[0].upper() + old_wrd[1:]
            else:
                return False
        #
        t = self.lexicon.inlexicon(r)
        i = self.flauberttok.ids(r, is_split_into_words=True)
        s, i_s = self.buildS(r)
        C, iC, iCC = self.buildC(idx)
        L, iL = self.buildL(idx)        
        #
        self.s[idx] = self.dword_create(r, s, i, t, i_s=i_s, E=E, iE=iE, C=C, iC=iC, iCC=iCC, L=L, iL=iL, p_cor=self.p_cor, p_lin=self.p_lin)
        #
        self.n_tokens_noised_with[E] += 1
        logging.debug('{}\t{} -> {}'.format(E,C,r))
        return True
    
    #########################################
    ### $SPLT ############################### r, s, i, t, E, C, iC, (plm, p)
    #########################################
    def inject_splt(self, idx):
        if idx >= len(self.s):
            return False
        if 'E' in self.s[idx]:
            return False
        if idx+1 >= len(self.s):
            return False
        if 'E' in self.s[idx+1]:
            return False
        if self.vocab_cor is not None and self.s[idx]['r'] not in self.vocab_cor: #i don't append words that i cannot generate
            return False
        #
        old_wrd = str(self.s[idx]['r'])
        r = str(self.s[idx]['r'] + self.s[idx+1]['r']) 
        i = self.flauberttok.ids(r, is_split_into_words=True)
        t = self.lexicon.inlexicon(r)
        #E = '$SPLT'
        E, iE = self.buildE('$SPLT')
        s, i_s = self.buildS(r)
        C, iC, iCC = self.buildC(idx)
        L, iL = self.buildL(idx)        
        #
        self.s[idx] = self.dword_create(r, s, i, t, i_s=i_s, E=E, iE=iE, C=C, iC=iC, iCC=iCC, L=L, iL=iL, p_cor=1.0, p_lin=self.p_lin)
        self.s.pop(idx+1) #deletes next
        #
        self.n_tokens_noised_with[E] += 1
        logging.debug('{}\t{} -> {}'.format(E,old_wrd,r))
        return True
        
    #########################################
    ### $MRGE ############################### r, s, i, t, E, C, iC, (plm, p)
    #########################################
    def inject_mrge(self, idx):
        if idx >= len(self.s):
            return False
        if 'E' in self.s[idx]:
            return False
        old_wrd = str(self.s[idx]['r'])
        minlen = 2 #minimum length of resulting spltted tokens
        if len(old_wrd) < 2*minlen:
            return False
        k = random.randint(minlen,len(old_wrd)-minlen)
        #
        ra = old_wrd[:k]
        ia = self.flauberttok.ids(ra,is_split_into_words=True)
        ta = self.lexicon.inlexicon(ra)
        #Ea = '$MRGE'
        Ea, iEa = self.buildE('$MRGE')
        sa, i_sa = self.buildS(ra)
        Ca, iCa, iCCa = self.buildC(idx)
        La, iLa = self.buildL(idx)        
        #
        rb = old_wrd[k:]
        ib = self.flauberttok.ids(rb,is_split_into_words=True)
        tb = self.lexicon.inlexicon(rb)
        #Eb = KEEP
        Eb, iEb = self.buildE(KEEP)
        sb, i_sb = self.buildS(rb)
        Cb, iCb, iCCb = self.buildC(idx)
        Lb, iLb = self.buildL(idx)        
        #
        self.s.pop(idx) #deletes idx
        self.s.insert(idx, self.dword_create(rb, sb, ib, tb, i_s=i_sb, E=Eb, iE=iEb, C=Cb, iC=iCb, iCC=iCCa, L=La, iL=iLa, p_cor=self.p_cor, p_lin=self.p_lin)) #inserted b
        self.s.insert(idx, self.dword_create(ra, sa, ia, ta, i_s=i_sa, E=Ea, iE=iEa, C=Ca, iC=iCa, iCC=iCCb, L=Lb, iL=iLb, p_cor=self.p_cor, p_lin=self.p_lin)) #inserted a b
        #
        self.n_tokens_noised_with[Ea] += 1
        logging.debug('{}\t{} -> {} {}'.format(Ea,old_wrd,ra,rb))
        return True
        
    #########################################
    ### $HYPH ############################### r, s, i, t, E, C, iC, (plm, p)
    #########################################
    def inject_hyph(self, idx):
        if idx >= len(self.s):
            return False
        if 'E' in self.s[idx]:
            return False
        old_wrd = str(self.s[idx]['r'])
        
        if old_wrd.count('-') == 1: ### HYPHm: Saint-Tropez -> Saint Tropez
            k = old_wrd.find('-')
            if k==0 or k==len(old_wrd)-1: ### hyphen not in the begining/end
                return False
            ra = old_wrd[:k]
            ia = self.flauberttok.ids(ra, is_split_into_words=True)
            ta = self.lexicon.inlexicon(ra)
            Ea = '$HYPHm'
            Ea, iEa = self.buildE('$HYPHm')
            sa, i_sa = self.buildS(ra)
            if sa not in ['x', 'X', 'Xx']:
                return False
            Ca, iCa, iCCa = self.buildC(idx)
            La, iLa = self.buildL(idx)        
            #
            rb = old_wrd[k+1:]
            ib = self.flauberttok.ids(rb, is_split_into_words=True)
            tb = self.lexicon.inlexicon(rb)
            #Eb = KEEP
            Eb, iEb = self.buildE(KEEP)
            sb, i_sb = self.buildS(rb)
            if sb not in ['x', 'X', 'Xx']:
                return False
            Cb, iCb, iCCb = self.buildC(idx)
            Lb, iLb = self.buildL(idx)        
            self.s.pop(idx) #deletes idx
            self.s.insert(idx, self.dword_create(rb, sb, ib, tb, i_s=i_sb, E=Eb, iE=iEb, C=Cb, iC=iCb, iCC=iCCb, L=Lb, iL=iLb, p_cor=self.p_cor, p_lin=self.p_lin)) #inserted b
            self.s.insert(idx, self.dword_create(ra, sa, ia, ta, i_s=i_sa, E=Ea, iE=iEa, C=Ca, iC=iCa, iCC=iCCa, L=La, iL=iLa, p_cor=self.p_cor, p_lin=self.p_lin)) #inserted a b
            #
            self.n_tokens_noised_with[Ea] += 1
            logging.debug('{}\t{} -> {} {}'.format(Ea,old_wrd,ra,rb))
            return True
        
        elif old_wrd.count('-') == 0: ### HYPHs: Saint Jacques => Saint - Jacques
            if idx == len(self.s)-1:
                return False
            if 'E' in self.s[idx+1]:
                return False
            if 'plm' not in self.s[idx]:
                return False
            if 'plm' not in self.s[idx+1]:
                return False
            if isinstance(self.s[idx]['plm'],list):
                logging.error(self.s[idx]['plm'])
                sys.exit()
            pos = self.s[idx]['plm'].split('|')[0]
            if pos not in ['NOM', 'ADJ']:
                return False
            pos = self.s[idx+1]['plm'].split('|')[0]
            if pos not in ['NOM', 'ADJ']:
                return False
            #
            ra = str(self.s[idx]['r'])
            ia = list(self.s[idx]['i'])
            ta = str(self.s[idx]['t'])
            Ea = KEEP
            Ea, iEa = self.buildE(KEEP)
            sa, i_sa = self.buildS(idx)
            Ca, iCa, iCCa = self.buildC(idx)
            La, iLa = self.buildL(idx)
            #
            rb = '-'
            ib = self.flauberttok.ids(rb, is_split_into_words=True)
            sb, i_sb = self.buildS(rb)
            tb = self.lexicon.inlexicon(rb)
            #Eb = '$HYPHs'
            Eb, iEb = self.buildE('$HYPHs')
            Cb = rb
            iCb = int(self.vocab_cor[rb]) if self.vocab_cor is not None else None
            iCCb = list(ib)
            Lb, iLb = None, None
            #
            rc = str(self.s[idx+1]['r'])
            ic = list(self.s[idx+1]['i'])
            tc = str(self.s[idx+1]['t'])
            Ec = KEEP
            Ec, iEc = self.buildE(KEEP)
            sc, i_sc = self.buildS(idx+1)
            Cc, iCc, iCCc = self.buildC(idx+1)
            Lc, iLc = self.buildL(idx+1)
            self.s.pop(idx) #deletes idx
            self.s.pop(idx) #deletes idx
            self.s.insert(idx, self.dword_create(rc, sc, ic, tc, i_s=i_sc, E=Ec, iE=iEc, C=Cc, iC=iCc, iCC=iCCc, L=Lc, iL=iLc, p_cor=self.p_cor, p_lin=self.p_lin)) #inserted c
            self.s.insert(idx, self.dword_create(rb, sb, ib, tb, i_s=i_sb, E=Eb, iE=iEb, C=Cb, iC=iCb, iCC=iCCb, L=Lb, iL=iLb, p_cor=self.p_cor, p_lin=self.p_lin)) #inserted b c
            self.s.insert(idx, self.dword_create(ra, sa, ia, ta, i_s=i_sa, E=Ea, iE=iEa, C=Ca, iC=iCa, iCC=iCCa, L=La, iL=iLa, p_cor=self.p_cor, p_lin=self.p_lin)) #inserted a b c
            #
            self.n_tokens_noised_with[Eb] += 1
            logging.debug('{}\t{} {} -> {} {} {}'.format(Eb,old_wrd,rc,ra,rb,rc))
            return True
        
        return False
    
if __name__ == '__main__': 

    parser = argparse.ArgumentParser()
    parser.add_argument('lex', type=str, default=None, help='Lexicon (pickle) file')
    parser.add_argument('--lemrules', type=str, default=None, help='LemRules file')
    parser.add_argument('--vocC', type=str, default=None, help='Vocabulary (corrections) file (None)')
    parser.add_argument('--vocF', type=str, default=None, help='Vocabulary (features) file (None)')
    parser.add_argument('--max_n', type=int, default=4, help='Maximum number of noises per sentence (4)')
    parser.add_argument('--max_r', type=float, default=0.3, help='Maximum ratio of noises/words per sentence (0.3)')
    parser.add_argument('--p_cor', type=float, default=0.5, help='probability of using cor layer when no err (0.5)')
    parser.add_argument('--p_lin', type=float, default=0.5, help='probability of using lin layer when no err (0.5)')
    #
    parser.add_argument('--w_phon', type=int, default=3, help='Weight of PHON noise [err/cor] (3)')
    parser.add_argument('--w_lemm', type=int, default=5, help='Weight of LEMM noise [err/cor] (5)')
    parser.add_argument('--w_apnd', type=int, default=1, help='Weight of APND noise [err/cor] (1)')
    parser.add_argument('--w_splt', type=int, default=1, help='Weight of SPLT noise [err/cor] (1)')
    parser.add_argument('--w_spel', type=int, default=1, help='Weight of SPEL noise [err/cor] (1)')
    parser.add_argument('--w_dele', type=int, default=1, help='Weight of DELE noise [err] (1)')
    parser.add_argument('--w_mrge', type=int, default=1, help='Weight of MRGE noise [err] (1)')
    parser.add_argument('--w_hyph', type=int, default=5, help='Weight of HYPH noise [err] (5)')
    parser.add_argument('--w_swap', type=int, default=1, help='Weight of SWAP noise [err] (1)')
    parser.add_argument('--w_case', type=int, default=1, help='Weight of CASE noise [err] (1)')
    #
    parser.add_argument('--seed', type=int, default=0, help='Seed for randomness (0)')
    parser.add_argument('--log', type=str, default="info", help='Log level (info)')
    args = parser.parse_args()
    create_logger(None, args.log)
    logging.info("Options = {}".format(args.__dict__))

    noises = []
    for k,v in args.__dict__.items():
        if k.startswith('w_'):
            noises.append([k[2:], v])
    
    noiser = Noiser(noises,args.lemrules,args.lex,args.vocC,args.vocF,args.max_n,args.max_r,args.p_cor,args.p_lin,args.seed)

    n_sents = 0
    tic = time.time()
    for l in sys.stdin:
        ldict = noiser(json.loads(l))
        print(json.dumps(ldict, ensure_ascii=False))
        n_sents += 1
        if n_sents%10000 == 0:
            noiser.debug(tic)
    logging.info('Done')
    noiser.debug(tic)
