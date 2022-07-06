import sys
import re
import pickle
import logging
from collections import defaultdict
from Utils import create_logger #, SEPAR1, SEPAR2

#DIACRITICS = 'aaàáâäeeéèêëiiíìîïooóòôöuuúùûü'

def rewrite_pos(pos,spos):
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
        pos = 'ART'
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
        sys.stderr.write('UNPARSED pos [{}] {}\n'.format(pos,l))
        pos = None
    return pos

def rewrite_lem(lem):
    if len(lem) > 3:
        lem = re.sub('^se ', '', lem)
    if len(lem) > 2:
        lem = re.sub('^s\'', '', lem)
    #lem = lem.replace('se ','').replace('s\'','')
    if len(lem) == 0 or ' ' in lem:
        #sys.stderr.write('UNPARSED lem [{}] {}\n'.format(lem,l))
        lem = None
    return lem

def rewrite_wrd(wrd):
    if len(wrd) > 3:
        wrd = re.sub('^se ', '', wrd)
    if len(wrd) > 2:
        wrd = re.sub('^s\'', '', wrd)
    if ' ' in wrd or len(wrd) == 0:
        #sys.stderr.write('UNPARSED wrd [{}] {}\n'.format(wrd,l))
        wrd = None
    return wrd

def rewrite_pho(pho):
    if pho == '':
        return ['-']
    phos = set()
    for p in pho.split(' OU '):
        p = p.replace(' ','').upper()
        p = p.replace('/','')
        if len(p):
            phos.add(p)
    return list(phos)

def rewrite_pers(pers):
    if pers == 'firstPerson':
        return '1'
    if pers == 'secondPerson':
        return '2'
    if pers == 'thirdPerson':
        return '3'
    if pers == '-':
        return '-'
    sys.stderr.write('pers: {}\n'.format(pers))
    return pers

def rewrite_nombre(nombre):
    if nombre == 'singular':
        return 'Sing'
    if nombre == 'plural':
        return 'Plur'
    if nombre == 'invariable':
        return 'Inv'
    if nombre == '-':
        return '-'
    sys.stderr.write('nombre: {}\n'.format(nombre))
    return nombre

def rewrite_genre(genre):
    if genre == 'feminine':
        return 'Fem'
    if genre == 'masculine':
        return 'Masc'
    if genre == 'invariable':
        return 'Inv'
    if genre == 'neuter':
        return '-'
    if genre == '-':
        return '-'
    sys.stderr.write('genre: {}\n'.format(genre))
    return genre

def add_flection(toks,txt2pos,txtpos2lem,lempos2txt,txtlempos2feats,txt2pho,pho2txt,base_lem,base_pos,base_spos,base_genre):
    #GRAPHIE;ID;NOMBRE;MODE;GENRE;TEMPS;PERSONNE;PHONÉTIQUE
    txt = rewrite_wrd(toks[0])
    if txt is None:
        logging.debug('filtered txt\t{}'.format(l))
        return False
    lem = rewrite_lem(base_lem)
    if lem is None:
        logging.debug('filtered lem\t{}'.format(l))
        return False
    pos = rewrite_pos(base_pos, base_spos)
    if pos is None:
        logging.debug('filtered pos\t{}'.format(l))
        return False
    nombre = rewrite_nombre(toks[2])
    mode   = toks[3]
    genre = rewrite_genre(toks[4]) if toks[4] != '-' else rewrite_genre(base_genre)
    temps  = toks[5]
    pers   = rewrite_pers(toks[6])
    phos = rewrite_pho(toks[7])

    txt2pos[txt].add(pos)
    txtpos2lem[txt+pos].add(lem)
    lempos2txt[lem+pos].add(txt)
    txtlempos2feats[txt+lem+pos].add( pos+'|'+lem+'|'+mode+'|'+temps+'|'+pers+'|'+nombre+'|'+genre )
    for pho in phos:
        #fdo.write('\t'.join([txt, pos, lem, mode, temps, pers, nombre, genre, pho])+'\n')
        fdo.write('\t'.join([txt, lem, pho, '|'.join([pos, mode, temps, pers, nombre, genre])])+'\n')
        if pho != '-':
            txt2pho[txt].add(pho)
            pho2txt[pho].add(txt)
    return True
            
################################################
### MAIN #######################################
################################################

if __name__ == '__main__':

    if len(sys.argv) != 2:
        sys.stderr.write('USAGE: {} path_to/Morphalou3.1_CSV.csv\n'.format(sys.argv[0]))
        sys.stderr.write('Creates path_to/Morphalou3.1_CSV.csv.{pickle,lex} files\n')
        sys.exit()
        
    create_logger(None, 'debug')
    fin = sys.argv[1]
    txt2pos = defaultdict(set)
    txtpos2lem = defaultdict(set)
    lempos2txt = defaultdict(set)
    txtlempos2feats = defaultdict(set)
    txt2pho = defaultdict(set)
    pho2txt = defaultdict(set)
        
    lem_str = ''
    pos_str = ''
    spos_str = ''
    genre_str = ''
    n = 0
    m = 0
    logging.info('Reading {}'.format(fin))
    with open(fin+'.lex', 'w') as fdo:
        with open(fin,'r') as fdi:
            for l in fdi:
                n += 1
                l = l.rstrip()
                toks = l.split(';')

                if len(toks) < 17 or toks[0] == 'LEMME' or toks[0] == 'GRAPHIE':
                    logging.debug('filtered\t{}'.format(l))
                    continue

                if toks[0] != '': ### this is a base form, keep some fields to be used with its flections
                    #GRAPHIE;ID;CATÉGORIE;SOUS CATÉGORIE;LOCUTION;GENRE;AUTRES LEMMES LIÉS;PHONÉTIQUE
                    lem_str = toks[0]
                    pos_str = toks[2]
                    spos_str = toks[3]
                    genre_str = toks[5]

                ### add entry
                if add_flection(toks[9:17],txt2pos,txtpos2lem,lempos2txt,txtlempos2feats,txt2pho,pho2txt,lem_str,pos_str,spos_str,genre_str):
                    m += 1
                    
    logging.info('Loaded Lexicon with {} entries, output {} entries'.format(n,m))
    logging.info('Saved {}'.format(fin+'.lex'))
    
    with open(fin+'.pickle', 'wb') as fdo:
        pickle.dump([txtpos2lem, txt2pos, txt2pho, lempos2txt, pho2txt, txtlempos2feats], fdo, protocol=-1)

    logging.info('Saved {}'.format(fin+'.pickle'))
        

