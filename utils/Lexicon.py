import sys
import time
import random
import logging
from collections import defaultdict
import pickle

class Lexicon():
    def __init__(self, f):
        with open(f, "rb") as fdi:
            #self.txtpos2lem, self.txt2pos, self.txt2pho, self.lempos2txt, self.pho2txt, self.txtlempos2feats = pickle.load(fdi)
            self.txtpos2lem, self.txt2pos, self.lempos2txt, self.txtlempos2feats = pickle.load(fdi)
            logging.info('Read {} with {} entries'.format(f,len(self.txt2pos)))

    def inlexicon(self, raw): #XXXIe, Mg, OTAN, NASA, Argentin, Saint-Siège, S.O.S., µA, A, B, C
        ### Find txt by applying some case-based modifs
        raw = raw.replace('’',"'")    
        if raw in self.txt2pos:
            return raw
        ### --vous (delete consecutive '-' if remaining exists)
        if len(raw) > 1 and raw[0] == '-':
            return self.inlexicon(raw[1:])
        ### all lowercase
        if raw.lower() in self.txt2pos:
            return raw.lower()
        ### upperacase letter: B, C
        if len(raw) == 1 and raw.upper() in self.txt2pos:
            return raw.upper()
        ### first uppercase rest lowercase: Mg, Saint
        if len(raw) > 1 and raw[0].upper() + raw[1:].lower() in self.txt2pos: #Ul+
            return raw[0].upper() + raw[1:].lower()
        ### all uppercase: OTAN
        if raw.upper() in self.txt2pos:
            return raw.upper() #U+
        ### not found
        logging.debug('raw not in lexicon SPACY: {}'.format(raw))
        return None
        
    def spacy2morphalou(self, txt, lem_spacy, pos_spacy, morph_spacy):
        ### find the most suitable pos in morphalou for given txt and spacy_pos
        pos = self.spacy2morphalou_pos(txt, pos_spacy, morph_spacy) #ADJ ADV ART ART:DEF ART:DEM ART:EXC ART:IND ART:POS CON INT NOM PRE PRO PRO:DEM PRO:INT PRO:PER PRO:POS PRO:REL VER
        if pos is None:
            logging.debug('pos not in lexicon: {} {} SPACY: {} {} {}'.format(txt, pos, lem_spacy, pos_spacy, morph_spacy))
            return None
        ### convert spacy feats to look like morphalou feats
        ### then, find lexicon lem/features that best match to txt/pos/spacy_feats_like_morphalou
        like_morphalou = self.spacy2morphalou_morph(txt, pos, morph_spacy) #['Sing', 'Fem', ...]
        best_plm = ''
        max_matchs = -1
        for lem in self.txtpos2lem[txt+pos]:
            if txt+lem+pos not in self.txtlempos2feats:
                continue
            for feats in self.txtlempos2feats[txt+lem+pos]:
                n_matchs = self.n_matchs(like_morphalou, feats)
                if n_matchs > max_matchs:
                    max_matchs = n_matchs
                    best_plm = feats
        if max_matchs == -1:
            logging.debug('feats not in lexicon: {} {} {} SPACY: {} {} {}'.format(txt, pos, like_morphalou, lem_spacy, pos_spacy, morph_spacy))
            return None
        return best_plm

    def n_matchs(self, morph, feats): #spacy morph converted to lexicon feats (Ex: ['Sing', 'Fem', ...]) VS feats of a lexicon entry (['indicative', 'present', '3', 'Sing', 'Fem'])
        n_matchs = 0
        for m in morph:
            if m in feats:
                n_matchs += 1
        return n_matchs

    def spacy2morphalou_pos(self, txt, pos_spacy, morph_spacy):
        if len(self.txt2pos[txt]) == 1:
            return list(self.txt2pos[txt])[0]
        
        if pos_spacy in ['VERB', 'AUX']:
            if 'VER' in self.txt2pos[txt]:
                return 'VER'

        elif pos_spacy in ['NOUN', 'PROPN']:
            if 'NOM' in self.txt2pos[txt]:
                return 'NOM'

        elif pos_spacy == 'ADJ':
            if 'ADJ' in self.txt2pos[txt]:
                return 'ADJ'

        elif pos_spacy == 'ADV':
            if 'ADV' in self.txt2pos[txt]:
                return 'ADV'

        elif pos_spacy == 'ADP':
            if 'PRE' in self.txt2pos[txt]:
                return 'PRE'

        elif pos_spacy == 'INTJ':
            if 'INT' in self.txt2pos[txt]:
                return 'INT'

        elif pos_spacy == 'PUNCT':
            if 'PUNCT' in self.txt2pos[txt]:
                return 'PUNCT'

        elif pos_spacy == 'SCONJ' or  pos_spacy == 'CCONJ':
            if 'CON' in self.txt2pos[txt]:
                return 'CON'

        elif pos_spacy == 'NUM':
            if 'ADJ' in self.txt2pos[txt]:
                return 'ADJ'
            
        elif pos_spacy == 'PRON': #Lexicon: qui	PRO:INT ou PRO:REL
            #if 'Person=' in morph_spacy and 'PRO:PER' in self.txt2pos[txt]:
            if 'PRO:PER' in self.txt2pos[txt]:
                return 'PRO:PER'
                
            #elif 'PronType=Rel' in morph_spacy and 'PRO:REL' in self.txt2pos[txt]:
            elif 'PRO:REL' in self.txt2pos[txt]:
                return 'PRO:REL'
                
            #elif 'PronType=Dem' in morph_spacy and 'PRO:DEM' in self.txt2pos[txt]:
            elif 'PRO:DEM' in self.txt2pos[txt]:
                return 'PRO:DEM'
                
            #elif 'PronType=Int' in morph_spacy and 'PRO:INT' in self.txt2pos[txt]:
            elif 'PRO:INT' in self.txt2pos[txt]:
                return 'PRO:INT'
                
            #elif 'NumType=Card' in morph_spacy and 'ADJ' in self.txt2pos[txt]:
            elif 'ADJ' in self.txt2pos[txt]:
                return 'ADJ'

            elif 'PRO' in self.txt2pos[txt]:
                return 'PRO'

        elif pos_spacy == 'DET': #pos=None not in lexicon:        plusieurs       plusieurs       DET     Number=Plur
            #if 'Definite=Def' in morph_spacy and 'ART:DEF' in self.txt2pos[txt]:
            if 'ART:DEF' in self.txt2pos[txt]:
                return 'ART:DEF'
                
            #elif 'Definite=Ind' in morph_spacy and 'ART:IND' in self.txt2pos[txt]:
            elif 'ART:IND' in self.txt2pos[txt]:
                return 'ART:IND'

            #elif 'Poss=Yes' in morph_spacy and 'ART:POS' in self.txt2pos[txt]:
            elif 'ART:POS' in self.txt2pos[txt]:
                return 'ART:POS'
                
            #elif 'PronType=Dem' in morph_spacy and 'ART:DEM' in self.txt2pos[txt]:
            elif 'ART:DEM' in self.txt2pos[txt]:
                return 'ART:DEM'
                
            #elif 'PronType=Int' in morph_spacy and 'ART:INT' in self.txt2pos[txt]:
            elif 'ART:INT' in self.txt2pos[txt]:
                return 'ART:INT'

            elif 'ART' in self.txt2pos[txt]:
                return 'ART'

        elif pos_spacy in ['X', 'SYM']:
            return None

        else:
            logging.error('UNPARSED POS: {}\t{}\t{}'.format(txt,pos_spacy,morph_spacy))

        return None

    def spacy2morphalou_morph(self, txt, pos, morph_spacy):
        morph = []
        for m in morph_spacy.split('|'):
            if m == '':
                pass

            elif m == 'Gender=Masc':
                morph.append('Masc')
            elif m == 'Gender=Fem':
                morph.append('Fem')

            elif m == 'Number=Sing':
                morph.append('Sing')
            elif m == 'Number=Plur':
                morph.append('Plur')

            elif m == 'Person=1':
                morph.append('1')
            elif m == 'Person=2':
                morph.append('2')
            elif m == 'Person=3':
                morph.append('3')

            elif m == 'Tense=Pres':
                morph.append('present')
            elif m == 'Tense=Past':
                morph.append('simplePast')
            elif m == 'Tense=Fut':
                morph.append('future')
            elif m == 'Tense=Imp':
                morph.append('imperfect')

            elif m == 'VerbForm=Part':
                morph.append('participle')
            elif m == 'VerbForm=Fin':
                pass
            elif m == 'VerbForm=Inf':
                morph.append('infinitive')

            elif m == 'Mood=Ind':
                morph.append('indicative')
            elif m == 'Mood=Sub':
                morph.append('subjunctive')
            elif m == 'Mood=Cnd':
                morph.append('conditional')
            elif m == 'Mood=Imp':
                morph.append('imperative')
                
            elif m == 'Voice=Pass':
                morph.append('Pass')

            elif m in ['Poss=Yes', 'Polarity=Neg', 'Definite=Ind', 'Definite=Def', 'PronType=Art', 'Reflex=Yes', 'PronType=Rel', 'PronType=Dem', 'PronType=Prs', 'PronType=Int', 'NumType=Card', 'NumType=Ord']:
                pass
            else:
                logging.error('UNPARSED MORPH: {}\t{}\t{}'.format(txt,pos,m))
        return morph

#    def homophones(self, txt):
#        txts = set()
#        for pho in self.txt2pho[txt]:
#            if pho == '-':
#                continue
#            for t in self.pho2txt[pho]:
#                if t != txt:
#                    txts.add(t)
#        if len(txts) == 0:
#            return None
#        txts = list(txts)
#        random.shuffle(txts)
#        return txts[0]
