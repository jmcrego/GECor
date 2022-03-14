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
        logging.debug('bad entry: {}'.format(l))
        return 0, None, None
    
    if len(toks[0]): 
        lem = toks[0]
        lem = lem.replace('se ','')
        lem = lem.replace('s\'','')
        pos = toks[2]
        if pos == 'Verbe':
            pos = 'VER'
        elif pos == 'Préposition':
            pos = 'PRE'
        elif pos == 'Adverbe':
            pos = 'ADV'
        elif pos == 'Conjonction':
            pos = 'CON'
        elif pos == 'Déterminant':
            pos = 'DET'
        elif pos == 'Interjection':
            pos = 'INT'
        elif pos == 'Pronom':
            pos = 'PRO'
        elif pos == 'Number':
            pos = 'NUM'
        elif pos.startswith('Adjectif'):
            pos = 'ADJ'
        elif pos.startswith('Nom'):
            pos = 'NOM'
        else:
            logging.info('unparsed pos: [{}]'.format(pos))

    lfeat = []
    wrd = toks[9]

    if ' ' in lem:
        logging.debug('bad lem: {}'.format(lem))
        return 0, lem, pos
    if ' ' in pos:
        logging.debug('bad entry: {}'.format(pos))
        return 0, lem, pos
    if ' ' in wrd:
        logging.debug('bad wrd: {}'.format(wrd))
        return 0, lem, pos
    
    if toks[12] != '-':
        vmode = toks[12]
        if vmode == 'indicative':
            vmode = 'ind'
        elif vmode == 'subjunctive':
            vmode = 'sub'
        elif vmode == 'conditional':
            vmode = 'cnd'
        elif vmode == 'imperative':
            vmode = 'imp'
        elif vmode == 'participle':
            vmode = 'par'
        elif vmode == 'infinitive':
            vmode = 'inf'
        else:
            logging.info('unparsed vmode: [{}]'.format(vmode))
        lfeat.append(vmode)
        
    if toks[13] != '-':
        gen = toks[13]
        if gen == 'masculine':
            lfeat.append('Masc')
        elif gen == 'feminine':
            lfeat.append('Fem')            
            
    if toks[14] != '-':
        temps = toks[14]
        if temps == 'simplePast':
            temps = 'simpas'
        elif temps == 'imperfect':
            temps = 'imp'
        elif temps == 'present':
            temps = 'pre'            
        elif temps == 'future':
            temps = 'fut'            
        elif temps == 'past':
            temps = 'pas'            
        else:
            logging.info('unparsed temps: [{}]'.format(temps))
        lfeat.append(temps)
        
    if toks[11] != '-':
        num = toks[11]
        if num == 'singular':
            lfeat.append('Sing')
        elif num == 'plural':
            lfeat.append('Plur')
            
    if toks[15] != '-':
        vpers = toks[15]
        lfeat.append(vpers)

    pho = toks[16]
    if 'OU' in pho:
        phos = pho.split(' OU ')
    else:
        phos = [pho]

    if wrd == 'FLEXION' or wrd == 'GRAPHIE':
        return 0, None, None

    if len(lfeat) == 0:
        lfeat = ['-']
    for pho in phos:
        pho = pho.replace(' ','')
        if len(pho):
            print('\t'.join([wrd,pho,lem,pos,':'.join(lfeat)]))
            nout += 1
        
    return nout, lem, pos

    
def parseLexique383(l,n):
    nout = 0
    if n == 0:
        return nout
    #ortho   phon    lemme   cgram   genre   nombre  freqlemfilms2   freqlemlivres   freqfilms2      freqlivres      infover
    toks = l.rstrip().split('\t')
    wrd = toks[0]
    if ' ' in wrd:
        logging.debug('bad wrd {}: {}'.format(n,l))
        return nout
    pho = toks[1]
    if ' ' in pho:
        logging.debug('bad pho {}: {}'.format(n,l))
        return nout
    lem = toks[2]
    if ' ' in lem:
        logging.debug('bad lem {}: {}'.format(n,l))
        return nout
    pos = toks[3]
    if ' ' in pos:
        logging.debug('bad pos {}: {}'.format(n,l))
        return nout
    if toks[4]=='m':
        gen = 'Masc'
    elif toks[4]=='f':
        gen = 'Fem'
    else:
        gen = None
    if toks[5]=='s':
        num = 'Sing'
    elif toks[5]=='p':
        num = 'Plur'
    else:
        num = None

    vers = toks[10] #imp:pre:2s;ind:pre:1s;ind:pre:3s;
    if len(vers):
        for lver in vers.split(';'):
            if len(lver) == 0:
                continue
            lfeat = lver.split(':')
            if gen is not None:
                lfeat.append(gen)
            if num is not None:
                lfeat.append(num)

            if '1s' in lver:
                lfeat.remove('1s')
                lfeat.append('Sing')
                lfeat.append('firstPerson')
            if '2s' in lver:
                lfeat.remove('2s')
                lfeat.append('Sing')
                lfeat.append('SecondPerson')
            if '3s' in lver:
                lfeat.remove('3s')
                lfeat.append('Sing')
                lfeat.append('thirdPerson')
            if '1p' in lver:
                lfeat.remove('1p')
                lfeat.append('Plur')
                lfeat.append('firstPerson')
            if '2p' in lver:
                lfeat.remove('2p')
                lfeat.append('Plur')
                lfeat.append('SecondPerson')
            if '3p' in lver:
                lfeat.remove('3p')
                lfeat.append('Plur')
                lfeat.append('thirdPerson')
                
            if len(lfeat) == 0:
                lfeat.append('-')
                            
            print('\t'.join([wrd,pho,lem,pos,':'.join(lfeat)]))
            nout += 1
    else:
        lfeat = []
        if gen is not None:
            lfeat.append(gen)
        if num is not None:
            lfeat.append(num)
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











    
