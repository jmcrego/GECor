#!/bin/bash

#APND, MRGE, SWAP, CASE SPLT, HYPH, DELE, LEMM, PHON, SPEL

gutenberg(){
    mkdir -p data/gutenberg
    ##########################
    ### download zim tools ###
    ##########################
    #wget https://download.openzim.org/release/zim-tools/zim-tools_linux-x86_64-3.1.1.tar.gz
    #tar xvzf zim-tools_linux-x86_64-3.1.1.tar.gz
    zim_dump=zim-tools_linux-x86_64-3.1.1/zimdump
    #######################################
    ### download gutenberg french books ### (http://download.kiwix.org/zim/gutenberg/)
    #######################################
    #wget http://download.kiwix.org/zim/gutenberg/gutenberg_fr_all_2022-05.zim
    zim_file=gutenberg_fr_all_2022-05.zim
    #############################
    ### dump all french books ###
    #############################
    #$zim_dump dump --dir data/gutenberg/dump $zim_file
    #python gutenberg_clean_dir.py data/gutenberg/dump/A data/gutenberg/books
    #cat data/gutenberg/books/* | python clean_sentences.py > data/gutenberg.fr 2> data/gutenberg.fr.log 
    #cat data/gutenberg.fr | sort | uniq > data/gutenberg.fr.uniq
}

morphalou(){
    python scripts/morphalou31_lexicon.py data/Morphalou3.1_CSV.csv
    python scripts/morphalou31_hphones.py data/Morphalou3.1_CSV.csv    
}

vocabs(){
    cat data/Morphalou3.1_CSV.csv.lexicon.lex | cut -f 3 | sort -u > data/vocabs.LFEATS
    python vocabs.py -o data/vocabs --lex data/Morphalou3.1_CSV.csv.lexicon.pickle data/BODY_2017_1_fr.txt data/gutenberg.fr.uniq
    cat data/vocabs.CORRECTIONS.freq  | perl -pe 's/^\d+\s+//' | grep -v -E "^[0-9]+$" | python scripts/discard_chars.py | grep -v "[[:punct:]][[:punct:]]" | head -50000 > data/vocabs.CORRECTIONS.50k
    cat data/vocabs.SHAPES.freq       | perl -pe 's/^\d+\s+//' > data/vocabs.SHAPES
}

#gutenberg
#morphalou
#vocabs &> log.vocabs &
flex=./data/Morphalou3.1_CSV.csv.lexicon.pickle
fpho=./data/Morphalou3.1_CSV.csv.hphones.pickle
flem=./data/lemrules.cfg
fcor=./data/vocabs.CORRECTIONS.50k
flin=./data/vocabs.LFEATS
ferr=./data/vocabs.ERRORS
fsha=./data/vocabs.SHAPES

fsrc=./data/HEADLINES_2018_8_fr.txt
#python spacyfy.py -l $flex $fsrc -n 4 -o $fsrc.json -voc_sha $fsha
#python noiser.py --lex $flex --pho $fpho --lem $flem --voc_cor $fcor --voc_lin $flin --voc_sha $fsha < $fsrc.json > $fsrc.json.noise

fsrc=./data/BODY_2017_1_fr.txt
#python spacyfy.py -l $flex $fsrc -n 32 -o $fsrc.json --voc_sha $fsha

fsrc=./data/gutenberg.fr.uniq
#python spacyfy.py -l $flex $fsrc -n 32 -o $fsrc.json --voc_sha $fsha

#############
### train ###
#############
exp=./exp
mkdir -p $exp

#python train.py --model kkmodel --train data/gutenberg.fr.uniq.noise --valid data/HEADLINES_2018_8_fr.txt.noise --err $ferr --cor $fcor --log_file kkmodel.log --cuda &
#python train.py --model kkmodel --train data/gutenberg.fr.uniq.noise --valid data/HEADLINES_2018_8_fr.txt.noise --err $ferr --cor $fcor --log_file kkmodel.log --cuda --accum_n_batchs 8 &
#python train.py --model models/kkmodel.00130000.pt --train kk.10k.json --err $ferr --cor $fcor


