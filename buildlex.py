import sys
import os
import time
import random
import logging
import argparse
from model.Utils import create_logger

def parseMorphalou31(l,lem=None,pos=None):
    nout = 0
    toks = l.split(';')
    
    if len(toks) < 17:
        logging.debug('BAD entry: {}'.format(l))
        return 0, None, None
    
    if len(toks[0]): 
        lem = toks[0].replace('se ','').replace('s\'','')
        pos = toks[2]
        spos = toks[3]
        #     40 Conjonction;coordination
        #    139 Conjonction;subordination
        #      4 Déterminant;défini
        #      5 Déterminant;démonstratif
        #      1 Déterminant;exclamatif
        #     40 Déterminant;indéfini
        #      7 Déterminant;possessif
        #      9 Pronom;démonstratif
        #     38 Pronom;indéfini
        #     12 Pronom;interrogatif
        #     42 Pronom;personnel
        #     11 Pronom;possessif
        #      9 Pronom;relatif
        #      2 Verbe;auxiliaire
        #     30 Verbe;défectif
        #     18 Verbe;impersonnel
        #  36523 Adjectif qualificatif
        #   4157 Adverbe
        #    179 Conjonction
        #     57 Déterminant
        #    422 Interjection
        #    198 Nombre
        # 102238 Nom commun
        #    258 Préposition
        #    121 Pronom
        #  14762 Verbe        
        if pos == 'Verbe':
            pos = 'VER'
        elif pos == 'Préposition':
            pos = 'PRE'
        elif pos == 'Adverbe':
            pos = 'ADV'
        elif pos == 'Conjonction':
            pos = 'CON'
        elif pos == 'Déterminant':
            if spos == 'défini':
                pos = 'ART:DEF'
            if spos == 'démonstratif':
                pos = 'ART:DEM'
            if spos == 'exclamatif':
                pos = 'ART:EXC'
            if spos == 'indéfini':
                pos = 'ART:IND'
            if spos == 'possessif':
                pos = 'ART:POS'
        elif pos == 'Interjection':
            pos = 'INT'
        elif pos == 'Pronom':
            pos = 'PRO'
            if spos == 'démonstratif':
                pos = 'PRO:DEM'
            elif spos == 'défini':
                pos = 'PRO:IND'
            elif spos == 'interrogatif':
                pos = 'PRO:INT'
            elif spos == 'personnel':
                pos = 'PRO:PER'
            elif spos == 'possessif':
                pos = 'PRO:POS'
            elif spos == 'relatif':
                pos = 'PRO:REL'                
        elif pos == 'Nombre':
            pos = 'ADJ'
        elif pos == 'Adjectif qualificatif':
            pos = 'ADJ'
        elif pos == 'Nom commun':
            pos = 'NOM'
        else:
            logging.debug('UNPARSED pos [{}] {}'.format(pos,lem))
            pos = None

    if pos == None or len(pos) == 0 or ' ' in pos:
        logging.debug('BAD pos [{}]'.format(pos))
        return 0, None, None

    if lem is None or len(lem) == 0 or ' ' in lem:
        logging.debug('BAD lem [{}]'.format(lem))
        return 0, None, None
    
    lfeat = []
    wrd = toks[9]

    if ' ' in wrd or len(wrd) == 0:
        logging.debug('BAD wrd: {}'.format(wrd))
        return 0, None, None
    
    if wrd == 'FLEXION' or wrd == 'GRAPHIE':
        return 0, None, None

    if toks[12] != '-':
        mood = toks[12]
        if mood == 'indicative':
            mood = 'Mood=Ind'
        elif mood == 'subjunctive':
            mood = 'Mood=Sub'
        elif mood == 'conditional':
            mood = 'Mood=Cnd'
        elif mood == 'imperative':
            mood = 'Mood=Imp'
        elif mood == 'participle':
            mood = 'VerbForm=Part'
        elif mood == 'infinitive':
            mood = 'VerbForm=Inf'
        else:
            logging.debug('UNPARSED mood [{}]'.format(mood))
        lfeat.append(mood)
        
    if toks[14] != '-':
        tense = toks[14]
        if tense == 'simplePast':
            tense = 'Tense=Past'
        elif tense == 'past':
            tense = 'Tense=Imp'
        elif tense == 'imperfect':
            tense = 'Tense=Imp'
        elif tense == 'present':
            tense = 'Tense=Pres'
        elif tense == 'future':
            tense = 'Tense=Fut'
        else:
            logging.debug('UNPARSED tense [{}]'.format(tense))
        lfeat.append(tense)
        
    if toks[13] != '-':
        gen = toks[13]
        if gen == 'masculine':
            lfeat.append('Gender=Masc')
        elif gen == 'feminine':
            lfeat.append('Gender=Fem')            
            
    if toks[11] != '-':
        number = toks[11]
        if number == 'singular':
            lfeat.append('Number=Sing')
        elif number == 'plural':
            lfeat.append('Number=Plur')
            
    if toks[15] != '-':
        person = toks[15]
        if person == 'firstPerson':
            lfeat.append('Person=1')
        elif person == 'secondPerson':
            lfeat.append('Person=2')
        elif person == 'thirdPerson':
            lfeat.append('Person=3')

    pho = toks[16]
    if 'OU' in pho:
        phos = pho.split(' OU ')
    else:
        phos = [pho]

    if len(lfeat) == 0:
        lfeat = ['-']
    for pho in phos:
        pho = pho.replace(' ','')
        if len(pho):
            print('\t'.join([wrd,pho,lem,pos,':'.join(lfeat)]))
            nout += 1
        
    return nout, lem, pos



    
def parseLexique383(l,n):
    #print(l.rstrip())
    nout = 0
    if n == 0:
        return nout
    #ortho   phon    lemme   cgram   genre   nombre  freqlemfilms2   freqlemlivres   freqfilms2      freqlivres      infover
    toks = l.rstrip().split('\t')
    wrd = toks[0]
    if ' ' in wrd or len(wrd) == 0:
        logging.debug('BAD wrd [{}]'.format(wrd))
        return nout
    pho = toks[1]
    if ' ' in pho or len(pho) == 0:
        logging.debug('BAD pho [{}]'.format(pho))
        return nout
    lem = toks[2]
    if ' ' in lem or len(lem) == 0:
        logging.debug('BAD lem [{}]'.format(lem))
        return nout
    pos = toks[3]
    #  26806 ADJ
    #      4 ADJ:dem
    #     36 ADJ:ind
    #      4 ADJ:int
    #    123 ADJ:num
    #     31 ADJ:pos
    #   1841 ADV
    #     10 ART:def
    #      4 ART:ind
    #     88 AUX
    #     35 CON
    #      1 LIA
    #  48287 NOM
    #    236 ONO
    #     80 PRE
    #     17 PRO:dem
    #     44 PRO:ind
    #     17 PRO:int
    #     53 PRO:per
    #     23 PRO:pos
    #     17 PRO:rel
    #  64929 VER
    if ' ' in pos or len(pos) == 0:
        logging.debug('BAD pos [{}]'.format(pos))
        return nout

    if pos == 'AUX':
        pos = 'VER'
    elif pos.startswith('ADJ'):
        pos = 'ADJ'

#    if pos == 'LIA' or pos == 'ONO':
#        logging.info('filtering pos [{}] {}'.format(pos,wrd))
#        return nout
    
    gen_num = []
    if toks[4]=='m':
        gen_num.append('Gender=Masc')
    elif toks[4]=='f':
        gen_num.append('Gender=Fem')
    if toks[5]=='s':
        gen_num.append('Number=Sing')
    elif toks[5]=='p':
        gen_num.append('Number=Plur')

    vers = toks[10] #imp:pre:2s;ind:pre:1s;ind:pre:3s;
    if len(vers):
        for lver in vers.split(';'):
            if len(lver) == 0:
                continue
            if lver.startswith('imp:'):
                lver = 'inp:' + lver[4:] ### imperatif

            lver = lver.split(':')
            lfeat = []
                
            if 'ind' in lver:
                lver.remove('ind')
                lfeat.append('Mood=Ind')
            elif 'sub' in lver:
                lver.remove('sub')
                lfeat.append('Mood=Sub')
            elif 'cnd' in lver:
                lver.remove('cnd')
                lfeat.append('Mood=Cnd')
            elif 'inp' in lver:
                lver.remove('inp')
                lfeat.append('Mood=Imp')

            if 'par' in lver:
                lver.remove('par')
                lfeat.append('VerbForm=Part')
            elif 'inf' in lver:
                lver.remove('inf')
                lfeat.append('VerbForm=Inf')
                
            if 'imp' in lver:
                lver.remove('imp')
                lfeat.append('Tense=Imp')
            elif 'pas' in lver:
                lver.remove('pas')
                lfeat.append('Tense=Past')
            elif 'fut' in lver:
                lver.remove('fut')
                lfeat.append('Tense=Fut')
            elif 'pre' in lver:
                lver.remove('pre')
                lfeat.append('Tense=Pres')
                
            if '1s' in lver:
                lver.remove('1s')
                lfeat.append('Number=Sing')
                lfeat.append('Person=1')
            elif '2s' in lver:
                lver.remove('2s')
                lfeat.append('Number=Sing')
                lfeat.append('Person=2')
            elif '3s' in lver:
                lver.remove('3s')
                lfeat.append('Number=Sing')
                lfeat.append('Person=3')
            elif '1p' in lver:
                lver.remove('1p')
                lfeat.append('Number=Plur')
                lfeat.append('Person=1')
            elif '2p' in lver:
                lver.remove('2p')
                lfeat.append('Number=Plur')
                lfeat.append('Person=2')
            elif '3p' in lver:
                lver.remove('3p')
                lfeat.append('Number=Plur')
                lfeat.append('Person=3')

            if len(lver):
                logging.error('missing feature {}'.format(lver))
                sys.exit()
                
            lfeat += gen_num
            
            if len(lfeat) == 0:
                lfeat.append('-')
                            
            print('\t'.join([wrd,pho,lem,pos,':'.join(lfeat)]))
            nout += 1
    else:
        lfeat = gen_num
        if len(lfeat) == 0:
            lfeat.append('-')

        print('\t'.join([wrd,pho,lem,pos,':'.join(lfeat)]))
        nout += 1

    return nout
        
######################################################################
### MAIN #############################################################
######################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='Lexicon file (required)', required=True)
    parser.add_argument('--log_file', type=str, default="stderr", help='Log file (stderr)')
    parser.add_argument('--log_level', type=str, default="info", help='Log level (info)')
    args = parser.parse_args()

    create_logger(args.log_file,args.log_level)
    logging.info("Options = {}".format(args.__dict__))
    tic = time.time()

    isLexique383 = 'Lexique383' in args.i

    lem=None
    pos=None
    with open(args.i, 'r') as fdi:
        nout = 0
        for n,l in enumerate(fdi):
            if isLexique383:
                nout += parseLexique383(l,n)
            else:
                no, lem, pos = parseMorphalou31(l,lem=lem,pos=pos)
                nout += no

                        
                
    toc = time.time()
    logging.info('Found {} entries ({:.2f} seconds)'.format(nout, toc-tic))











    
