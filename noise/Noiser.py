import sys
import json
import time
import spacy
import random
import logging
from collections import defaultdict
from utils.FlaubertTok import FlaubertTok
from utils.Lexicon import Lexicon
from utils.Utils import MAX_IDS_LEN, KEEP, shape_of_word, reshape, PUNCTUATION
from noise.LemRules import LemRules
from noise.Misspell import Misspell
from noise.Homophones import Homophones
from model.Vocab import Vocab


class Noiser():
    def __init__(self, noises, lem, lex, pho, voce, vocc, vocl, vocs, voci, max_noises=0, max_ratio=0.5, p_cor=0.0, p_lng=0.0, seed=0):
        if seed:
            random.seed(seed)
        self.flauberttok = FlaubertTok(max_ids_len=MAX_IDS_LEN)
        self.lexicon = Lexicon(lex)
        self.lemrules = LemRules(lem,self.lexicon) if lem is not None else None
        self.homophones = Homophones(pho)
        self.vocab_err = Vocab(voce) if voce is not None else None
        self.vocab_cor = Vocab(vocc) if vocc is not None else None
        self.vocab_lng = Vocab(vocl) if vocl is not None else None
        self.vocab_sha = Vocab(vocs) if vocs is not None else None
        self.vocab_inl = Vocab(voci) if voci is not None else None
        self.misspell = Misspell()
        self.noises = [x[0] for x in noises]
        self.wnoises = [x[1] for x in noises]
        self.max_noises = max_noises
        self.max_ratio = max_ratio
        self.p_cor = p_cor
        self.p_lng = p_lng
        self.n_sents = 0
        self.n_tokens = 0
        self.n_tokens_noised_with = defaultdict(int) #n_tokens_noised_with['$APND'] += 1
        self.n_sents_with_n_noises = defaultdict(int) #n_sents_with_n_noises[3] += 1
        self.n_tokens_predict = defaultdict(int) #n_sents_with_n_noises['cor'] += 1
        logging.info("Built Noiser")

    def __call__(self, ldict):
        self.s = ldict
        self.n_sents += 1
        self.n_tokens += len(self.s)
        n_noises_sentence = 0
        n_noises_toinject = random.randint(0, min(self.max_noises, int(len(self.s)*self.max_ratio)))
        #logging.debug("BEFORE: {}".format([d['raw'] for d in ldict]))
        for _ in range(n_noises_toinject*2): ### try this maximum number of times before stopping
            if n_noises_sentence >= n_noises_toinject:
                break
            idxs = [idx for idx in range(len(self.s)) if 'ierr' not in self.s[idx]]
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
            elif noise == 'hyph': ### splt/mrge words using hyph and add $HYPs/$HYPm
                n_noises_sentence += self.inject_hyph(idx)
            elif noise == 'swap': ### swap current/next words and add $SWAP error to first
                n_noises_sentence += self.inject_swap(idx)
            elif noise == 'case': ### change case of current word and add $CAS1/$CASn
                n_noises_sentence += self.inject_case(idx)
        self.n_sents_with_n_noises[n_noises_sentence] += 1
        for idx in range(len(self.s)):
            if 'ierr' in self.s[idx]:
                continue
            self.n_tokens_predict['total'] += 1
            lng, ilng = None, None
            cor, icor, iCOR = None, None, None
            
            if 'plm' in self.s[idx] and random.random() < self.p_lng:
                lng, ilng = self.buildLng(idx)
                if lng is not None:
                    self.n_tokens_predict['lng'] += 1

            if self.s[idx]['raw'] in self.vocab_cor and random.random() < self.p_cor:
                cor, icor, iCOR = self.buildCor_from(idx)
                if cor is not None:
                    self.n_tokens_predict['cor'] += 1
                    
            if lng is not None or cor is not None:
                err, ierr = self.buildErr(KEEP)
                raw, iraw = self.buildRaw_as(idx)
                shp, ishp = self.buildShp_as(idx)
                lex, ilex = self.buildLex_as(idx)
                self.s[idx] = self.dword(raw, iraw, shp, ishp, lex, ilex, err=err, ierr=ierr, cor=cor, icor=icor, iCOR=iCOR, lng=lng, ilng=ilng)
                
        #logging.debug("AFTER : {}".format([d['raw'] for d in ldict]))
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
        for noise,n in sorted(self.n_tokens_noised_with.items(), key=lambda kv: kv[1], reverse=True):
            logging.info('\t{}   \t{} toks ({:.2f}%)'.format(noise,n,100.0*n/n_tokens_noised))

        if self.n_tokens_predict['total'] > 0:
            logging.info('Unnoised {} toks'.format(self.n_tokens_predict['total']))
            logging.info('\tadded {} pred \t{} toks ({:.2f}%)'.format('COR', self.n_tokens_predict['cor'], 100.0*self.n_tokens_predict['cor']/self.n_tokens_predict['total']))
            logging.info('\tadded {} pred \t{} toks ({:.2f}%)'.format('LNG', self.n_tokens_predict['lng'], 100.0*self.n_tokens_predict['lng']/self.n_tokens_predict['total']))

    #########################################
    ### $APND ###############################
    #########################################
    def inject_apnd(self, idx):
        if idx >= len(self.s):
            return False
        if 'ierr' in self.s[idx]:
            return False
        if idx+1 >= len(self.s):
            return False
        if 'ierr' in self.s[idx+1]:
            return False
        if 'plm' not in self.s[idx+1] and self.s[idx+1]['raw'] not in PUNCTUATION: #i don't append words without linguistic annotations unless punctuation
            return False
        if self.s[idx+1]['raw'] not in self.vocab_cor: #i don't append words that i cannot generate
            return False
        if 'plm' in self.s[idx+1] and self.s[idx+1]['plm'].split('|')[0] in ['VER', 'ADJ', 'NOM']: #i don't append open class words
            return False
        #
        raw, iraw = self.buildRaw_as(idx)
        shp, ishp = self.buildShp_as(idx)
        lex, ilex = self.buildLex_as(idx)
        err, ierr = self.buildErr('$APND')
        cor, icor, iCOR = self.buildCor_from(idx+1)
        ###lng, ilng = self.buildLng_from(idx)
        self.s[idx] = self.dword(raw, iraw, shp, ishp, lex, ilex, err=err, ierr=ierr, cor=cor, icor=icor, iCOR=iCOR)
        self.s.pop(idx+1) #delete idx+1
        #
        self.n_tokens_noised_with[err] += 1
        logging.debug('{}\t{}'.format(err,cor))
        return True

    #########################################
    ### $DELE ###############################
    #########################################
    def inject_dele(self, idx):
        if idx >= len(self.s):
            return False
        if 'ierr' in self.s[idx]:
            return False
        #
        raw, iraw = self.buildRaw_as(idx)
        shp, ishp = self.buildShp_as(idx)
        lex, ilex = self.buildLex_as(idx)
        err, ierr = self.buildErr('$DELE')
        err2, ierr2 = self.buildErr(KEEP)
        #cor, icor, iCOR = self.buildCor_from(idx)
        ###lng, ilng = self.buildLng_from(idx)
        self.s.pop(idx) #replaced by the next two words
        self.s.insert(idx, self.dword(raw, iraw, shp, ishp, lex, ilex, err=err,  ierr=ierr)) #must be deleted
        self.s.insert(idx, self.dword(raw, iraw, shp, ishp, lex, ilex, err=err2, ierr=ierr2)) #must be kept
        #
        self.n_tokens_noised_with[err] += 1
        logging.debug('{}\t{}'.format(err,raw))
        return True

    #########################################
    ### $SWAP ############################### r, s, i, t, E, C, iC, (plm, p)
    #########################################
    def inject_swap(self, idx):
        if idx >= len(self.s):
            return False
        if 'ierr' in self.s[idx]:
            return False
        if idx+1 >= len(self.s):
            return False
        if 'ierr' in self.s[idx+1]:
            return False
        #
        rawa, irawa = self.buildRaw_as(idx)
        shpa, ishpa = self.buildShp_as(idx)
        lexa, ilexa = self.buildLex_as(idx)
        erra, ierra = self.buildErr(KEEP)
        #cora, icora, iCORa = self.buildCor_from(idx)
        #lnga, ilnga = self.buildLng_from(idx)
        #
        rawb, irawb = self.buildRaw_as(idx+1)
        shpb, ishpb = self.buildShp_as(idx+1)
        lexb, ilexb = self.buildLex_as(idx+1)
        errb, ierrb = self.buildErr('$SWAP')
        #corb, icorb, iCORb = self.buildCor_from(idx+1)
        #lngb, ilngb = self.buildLng_from(idx+1)
        #
        self.s.pop(idx) #deletes a
        self.s.pop(idx) #deletes b
        self.s.insert(idx, self.dword(rawa, irawa, shpa, ishpa, lexa, ilexa, err=erra,  ierr=ierra)) #inserts a
        self.s.insert(idx, self.dword(rawb, irawb, shpb, ishpb, lexb, ilexb, err=errb,  ierr=ierrb)) #inserts b a
        #
        self.n_tokens_noised_with[errb] += 1
        logging.debug('{}\t{} <-> {}'.format(errb,rawb,rawa))
        return True

    #########################################
    ### $LEMM ###############################
    #########################################
    def inject_lemm(self, idx):
        if self.lemrules is None:
            return False
        if idx >= len(self.s):
            return False
        if 'ierr' in self.s[idx]:
            return False
        if 'plm' not in self.s[idx]:
            return False
        if self.vocab_cor is not None and self.s[idx]['raw'] not in self.vocab_cor: #i don't append words that i cannot generate
            return False
        lex, new_plm = self.lemrules(self.s[idx])
        old_plm = str(self.s[idx]['plm'])
        if lex is None:
            return False
        #
        raw = reshape(lex,self.s[idx]['shp'])
        iraw = self.flauberttok.ids(raw, is_split_into_words=True)
        lex, ilex = self.buildLex(raw)
        err, ierr = self.buildErr('$LEMM')
        shp, ishp = self.buildShp(raw)
        cor, icor, iCOR =  self.buildCor_from(idx)
        lng, ilng =  self.buildLng(idx)
        #
        self.s[idx] = self.dword(raw, iraw, shp, ishp, lex, ilex, err=err,  ierr=ierr,  cor=cor, icor=icor, iCOR=iCOR, lng=lng, ilng=ilng)
        #
        self.n_tokens_noised_with[err] += 1
        logging.debug('{}\t{} -> {}\t{} -> {}'.format(err,cor,raw,old_plm,new_plm))
        return True

    #########################################
    ### $SPEL ###############################
    #########################################
    def inject_spel(self, idx):
        if idx >= len(self.s):
            return False
        if 'ierr' in self.s[idx]:
            return False
        if self.vocab_cor is not None and self.s[idx]['raw'] not in self.vocab_cor: #i don't append words that i cannot generate
            return False
        if not self.s[idx]['raw'].isalpha() and self.s[idx]['raw'] not in PUNCTUATION: ### alphabetic or punctuation (attention with hyphens)
            return False
        raw = self.misspell(self.s[idx]['raw'])
        if raw is None:
            return False
        #
        iraw = self.flauberttok.ids(raw,is_split_into_words=True)
        lex, ilex = self.buildLex(raw)
        err, ierr = self.buildErr('$SPEL')
        shp, ishp = self.buildShp(raw)
        cor, icor, iCOR = self.buildCor_from(idx)
        #lng, ilng = self.buildLng(idx)        
        #
        self.s[idx] = self.dword(raw, iraw, shp, ishp, lex, ilex, err=err,  ierr=ierr,  cor=cor, icor=icor, iCOR=iCOR)
        #
        self.n_tokens_noised_with[err] += 1
        logging.debug('{}\t{} -> {}'.format(err,cor,raw))
        return True

    #########################################
    ### $PHON ###############################
    #########################################
    def inject_phon(self, idx):
        if idx >= len(self.s):
            return False
        if 'ierr' in self.s[idx]:
            return False
        if self.vocab_cor is not None and self.s[idx]['raw'] not in self.vocab_cor: #i don't append words that i cannot generate
            return False
        if self.s[idx]['lex'] is None: #not in lexicon
            return False
        lex = self.homophones(self.s[idx]['lex'])
        if lex is None:
            return False
        #
        raw = reshape(lex,self.s[idx]['shp'])
        iraw = self.flauberttok.ids(raw, is_split_into_words=True)
        lex, ilex = self.buildLex(raw)
        shp, ishp = self.buildShp(raw)
        err, ierr = self.buildErr('$PHON')
        cor, icor, iCOR = self.buildCor_from(idx)
        #lng, ilng = self.buildLng(idx)
        #
        self.s[idx] = self.dword(raw, iraw, shp, ishp, lex, ilex, err=err,  ierr=ierr,  cor=cor, icor=icor, iCOR=iCOR)
        #
        self.n_tokens_noised_with[err] += 1
        logging.debug('{}\t{} -> {}'.format(err,cor,raw))
        return True
        
    #########################################
    ### $SPLT ###############################
    #########################################
    def inject_splt(self, idx):
        if idx >= len(self.s):
            return False
        if 'ierr' in self.s[idx]:
            return False
        if idx+1 >= len(self.s):
            return False
        if 'ierr' in self.s[idx+1]:
            return False
        if self.vocab_cor is not None and self.s[idx]['raw'] not in self.vocab_cor: #i don't append words that i cannot generate
            return False
        #
        old_wrd = str(self.s[idx]['raw'])
        raw = str(self.s[idx]['raw'] + self.s[idx+1]['raw']) 
        iraw = self.flauberttok.ids(raw, is_split_into_words=True)
        lex, ilex = self.buildLex(raw)
        shp, ishp = self.buildShp(raw)
        err, ierr = self.buildErr('$SPLT')
        cor, icor, iCOR = self.buildCor_from(idx)
        #lng, ilng = self.buildLng(idx)        
        #
        self.s[idx] = self.dword(raw, iraw, shp, ishp, lex, ilex, err=err,  ierr=ierr,  cor=cor, icor=icor, iCOR=iCOR)
        self.s.pop(idx+1) #deletes next
        #
        self.n_tokens_noised_with[err] += 1
        logging.debug('{}\t{} -> {}'.format(err,old_wrd,raw))
        return True
        
    #########################################
    ### $MRGE ###############################
    #########################################
    def inject_mrge(self, idx):
        if idx >= len(self.s):
            return False
        if 'err' in self.s[idx]:
            return False
        old_wrd = str(self.s[idx]['raw'])
        minlen = 2 #minimum length of resulting spltted tokens
        if len(old_wrd) < 2*minlen:
            return False
        k = random.randint(minlen,len(old_wrd)-minlen)
        #
        rawa = old_wrd[:k]
        irawa = self.flauberttok.ids(rawa,is_split_into_words=True)
        lexa, ilexa = self.buildLex(rawa)
        shpa, ishpa = self.buildShp(rawa)
        erra, ierra = self.buildErr('$MRGE')
        #cora, icora, iCORa = self.buildCor_from(idx)
        #lnga, ilnga = self.buildLng(idx)
        #
        rawb = old_wrd[k:]
        irawb = self.flauberttok.ids(rawb,is_split_into_words=True)
        lexb, ilexb = self.buildLex(rawb)
        shpb, ishpb = self.buildShp(rawb)
        errb, ierrb = self.buildErr(KEEP)
        #corb, icorb, iCORb = self.buildCor_from(idx)
        #lngb, lngb = self.buildLng(idx)        
        #
        self.s.pop(idx) #deletes idx
        self.s.insert(idx, self.dword(rawb, irawb, shpb, ishpb, lexb, ilexb, err=errb,  ierr=ierrb)) #inserted b
        self.s.insert(idx, self.dword(rawa, irawa, shpa, ishpa, lexa, ilexa, err=erra,  ierr=ierra)) #inserted a b
        #
        self.n_tokens_noised_with[erra] += 1
        logging.debug('{}\t{} -> {} {}'.format(erra,old_wrd,rawa,rawb))
        return True
        
    #########################################
    ### $CASE ###############################
    #########################################
    def inject_case(self, idx):
        if idx >= len(self.s):
            return False
        if 'ierr' in self.s[idx]:
            return False
        if self.vocab_cor is not None and self.s[idx]['raw'] not in self.vocab_cor: #i don't append words that i cannot generate
            return False
        #
        old_wrd = str(self.s[idx]['raw'])
        old_shape = str(self.s[idx]['shp'])
        if len(old_shape) == 1: ### all lowercased or uppercased
            if len(old_wrd) == 1:
                err, ierr = self.buildErr('$CASn')
                if old_shape == 'X':
                    raw = old_wrd.lower()
                elif old_shape == 'x':
                    raw = old_wrd.upper()
                else:
                    return False
            else: #len(old_wrd)>1
                if random.random() < 0.5: #CASn
                    err, ierr = self.buildErr('$CASn')
                    if old_shape == 'X':
                        raw = old_wrd.lower()
                    elif old_shape == 'x':
                        raw = old_wrd.upper()
                    else:
                        return False
                else: #CAS1
                    err, ierr = self.buildErr('$CAS1')
                    if old_shape == 'X':
                        raw = old_wrd[0].lower() + old_wrd[1:]
                    elif old_shape == 'x':
                        raw = old_wrd[0].upper() + old_wrd[1:]
                    else:
                        return False
        else: #len(txt_shape) > 1 (i can only do CAS1)
            err, ierr = self.buildErr('$CAS1')
            if old_shape.startswith('X'):
                raw = old_wrd[0].lower() + old_wrd[1:]
            elif old_shape.startswith('x'):
                raw = old_wrd[0].upper() + old_wrd[1:]
            else:
                return False
        #
        iraw = self.flauberttok.ids(raw, is_split_into_words=True)
        lex, ilex = self.buildLex(raw)
        shp, ishp = self.buildShp(raw)
        #cor, icor, iCOR = self.buildCor_from(idx)
        #lng, ilng = self.buildLng(idx)
        #
        self.s[idx] = self.dword(raw, iraw, shp, ishp, lex, ilex, err=err,  ierr=ierr)
        #
        self.n_tokens_noised_with[err] += 1
        logging.debug('{}\t{} -> {}'.format(err,old_wrd,raw))
        return True
    
    #########################################
    ### $HYPH ###############################
    #########################################
    def inject_hyph(self, idx):
        if idx >= len(self.s):
            return False
        if 'ierr' in self.s[idx]:
            return False
        old_wrd = str(self.s[idx]['raw'])
        
        if old_wrd.count('-') == 1: ### HYPm: Saint-Tropez -> Saint Tropez
            k = old_wrd.find('-')
            if k==0 or k==len(old_wrd)-1: ### hyphen cannot be in the begining/end
                return False
            rawa = old_wrd[:k]
            irawa = self.flauberttok.ids(rawa, is_split_into_words=True)
            lexa, ilexa = self.buildLex(rawa)
            erra, ierra = self.buildErr('$HYPm')
            shpa, ishpa = self.buildShp(rawa)
            if shpa not in ['x', 'X', 'Xx']:
                return False
            #cora, icora, iCCORa = self.buildCor_from(idx)
            #lnga, ilnga = self.buildLng(idx)
            #
            rawb = old_wrd[k+1:]
            irawb = self.flauberttok.ids(rawb, is_split_into_words=True)
            lexb, ilexb = self.buildLex(rawb)
            errb, ierrb = self.buildErr(KEEP)
            shpb, ishpb = self.buildShp(rawb)
            if shpb not in ['x', 'X', 'Xx']:
                return False
            #corb, icorb, iCORb = self.buildCor_from(idx)
            #lngb, ilngb = self.buildLng(idx)
            #
            self.s.pop(idx) #deletes idx
            self.s.insert(idx, self.dword(rawb, irawb, shpb, ishpb, lexb, ilexb, err=errb,  ierr=ierrb)) #inserted b
            self.s.insert(idx, self.dword(rawa, irawa, shpa, ishpa, lexa, ilexa, err=erra,  ierr=ierra)) #inserted a b
            #
            self.n_tokens_noised_with[erra] += 1
            logging.debug('{}\t{} -> {} {}'.format(erra,old_wrd,rawa,rawb))
            return True
        
        elif old_wrd.count('-') == 0: ### HYPs: Saint Jacques => Saint - Jacques
            if idx == len(self.s)-1:
                return False
            if 'ierr' in self.s[idx+1]:
                return False
            if 'plm' not in self.s[idx]:
                return False
            if 'plm' not in self.s[idx+1]:
                return False
            if self.s[idx]['plm'].split('|')[0] not in ['NOM', 'ADJ']:
                return False
            if self.s[idx+1]['plm'].split('|')[0] not in ['NOM', 'ADJ']:
                return False
            #
            rawa, irawa = self.buildRaw_as(idx)
            lexa, ilexa = self.buildLex_as(idx)
            erra, ierra = self.buildErr(KEEP)
            shpa, ishpa = self.buildShp_as(idx)
            #cora, icora, iCORa = self.buildCor_from(idx)
            #lnga, ilnga = self.buildLng(idx)
            #
            rawb = '-'
            irawb = self.flauberttok.ids(rawb, is_split_into_words=True)
            shpb, ishpb = self.buildShp(rawb)
            lexb, ilexb = self.buildLex(rawb)
            errb, ierrb = self.buildErr('$HYPs')
            #
            rawc, irawc = self.buildRaw_as(idx+1)
            lexc, ilexc = self.buildLex_as(idx+1)
            errc, ierrc = self.buildErr(KEEP)
            shpc, ishpc = self.buildShp_as(idx+1)
            #corc, icorc, iCORc = self.buildCor_from(idx+1)
            #lngc, ilngc = self.buildLng(idx+1)
            self.s.pop(idx) #deletes idx
            self.s.pop(idx) #deletes idx
            self.s.insert(idx, self.dword(rawc, irawc, shpc, ishpc, lexc, ilexc, err=errc,  ierr=ierrc)) #inserted c
            self.s.insert(idx, self.dword(rawb, irawb, shpb, ishpb, lexb, ilexb, err=errb,  ierr=ierrb)) #inserted b c
            self.s.insert(idx, self.dword(rawa, irawa, shpa, ishpa, lexa, ilexa, err=erra,  ierr=ierra)) #inserted a b c
            #
            self.n_tokens_noised_with[errb] += 1
            logging.debug('{}\t{} {} -> {} {} {}'.format(errb,rawa,rawc,rawa,rawb,rawc))
            return True
        
        return False
    
    def buildRaw_as(self, idx):
        raw = str(self.s[idx]['raw'])
        iraw = list(self.s[idx]['iraw'])
        return raw, iraw

    def buildShp_as(self, idx):
        shp = str(self.s[idx]['shp'])
        ishp = int(self.s[idx]['ishp'])
        return shp, ishp

    def buildShp(self, raw):
        shp = shape_of_word(raw) #compute new shape
        ishp = int(self.vocab_sha[shp]) #if self.vocab_sha is not None else None
        return shp, ishp

    def buildLex_as(self, idx):
        lex = str(self.s[idx]['lex'])
        ilex = int(self.s[idx]['ilex'])
        return lex, ilex

    def buildLex(self, raw):
        lex = self.lexicon.inlexicon(raw)
        ilex = lex is not None
        ilex = int(self.vocab_inl[ilex]) #if self.vocab_inl is not None else None
        return lex, ilex
    
    def buildErr(self, err):
        err = str(err)
        ierr = int(self.vocab_err[err]) #if self.vocab_err is not None else None
        return err, ierr

    def buildLng(self, idx):
        if 'plm' not in self.s[idx] or self.s[idx]['plm'] is None:
            #logging.info('not plm for idx={} {}'.format(idx,self.s[idx]['raw']))
            return None, None
        lng = self.s[idx]['plm'].split('|')
        lng.pop(1) #remove lemma
        lng = str('|'.join(lng))
        ilng = int(self.vocab_lng[lng]) #if self.vocab_lng is not None else None
        return lng, ilng
            
    def buildCor_from(self, idx):
        cor = str(self.s[idx]['raw'])
        icor = self.vocab_cor[cor]
        iCOR = list(self.s[idx]['iraw'])
        return cor, icor, iCOR
        
    def dword(self, raw, iraw, shp, ishp, lex, ilex, err=None, ierr=None, cor=None, icor=None, iCOR=None, lng=None, ilng=None):
        d = {'raw':str(raw), 'iraw':list(iraw), 'shp':str(shp), 'ishp':int(ishp), 'lex':str(lex), 'ilex':str(ilex)}
        if err is not None and ierr is not None:
            d['err'] = str(err)
            d['ierr'] = int(ierr)
        if cor is not None and icor is not None and iCOR is not None:
            d['cor'] = str(cor)
            d['icor'] = int(icor)
            d['iCOR'] = list(iCOR)
        if lng is not None and ilng is not None:
            d['lng'] = str(lng)
            d['ilng'] = int(ilng)
        return d

if __name__ == '__main__': 

    parser = argparse.ArgumentParser()
    parser.add_argument('lex', type=str, default=None, help='Lexicon (pickle) file')
    parser.add_argument('--lemrules', type=str, default=None, help='LemRules file')
    parser.add_argument('--vocC', type=str, default=None, help='Vocabulary (corrections) file (None)')
    parser.add_argument('--vocF', type=str, default=None, help='Vocabulary (features) file (None)')
    parser.add_argument('--max_n', type=int, default=4, help='Maximum number of noises per sentence (4)')
    parser.add_argument('--max_r', type=float, default=0.3, help='Maximum ratio of noises/words per sentence (0.3)')
    parser.add_argument('--p_cor', type=float, default=0.5, help='probability of using cor layer when no err (0.5)')
    parser.add_argument('--p_lng', type=float, default=0.5, help='probability of using lng layer when no err (0.5)')
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
    
    noiser = Noiser(noises,args.lemrules,args.lex,args.vocC,args.vocF,args.max_n,args.max_r,args.p_cor,args.p_lng,args.seed)

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
