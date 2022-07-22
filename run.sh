#!/bin/bash

dictates(){
    cp ../gramerco/test-dictates/dicts.cor examples/dicts.cor.raw
    cp ../gramerco/test-dictates/dicts.err examples/dicts.err.raw
    cat examples/dicts.err.raw | iconv -t UTF-8 | tr "\n" " " | perl -pe 's/\s*###/\n/g' | perl -pe 's/\x{c2}\x{a0}/ /g' | perl -pe "s/ ' /'/g" | perl -pe 's/  +/ /g' | perl -pe 's/« /«/g' | perl -pe 's/ »/»/g' > examples/dicts.err
    cat examples/dicts.cor.raw | iconv -t UTF-8 | tr "\n" " " | perl -pe 's/\s*###/\n/g' | perl -pe 's/\x{c2}\x{a0}/ /g' | perl -pe "s/ ' /'/g" | perl -pe 's/  +/ /g' | perl -pe 's/« /«/g' | perl -pe 's/ »/»/g' > examples/dicts.cor
}

news-crawl(){
    exit
    #wget -O data/news.2021.fr.shuffled.deduped.gz https://data.statmt.org/news-crawl/fr/news.2021.fr.shuffled.deduped.gz
    #wget -O data/news.2007.fr.shuffled.deduped.gz https://data.statmt.org/news-crawl/fr/news.2007.fr.shuffled.deduped.gz
    #zcat data/news.2021.fr.shuffled.deduped.gz | python scripts/sentencesegmenter.py | sort -u | grep "^[\'\"\«A-Z]" | grep "[\.\,\;\?\!\'\»\"\:]$" > data/news.2021.fr.txt
    #zcat data/news.2007.fr.shuffled.deduped.gz | python scripts/sentencesegmenter.py | sort -u | grep "^[\'\"\«A-Z]" | grep "[\.\,\;\?\!\'\»\"\:]$" > data/news.2007.fr.txt
}

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
    #python scripts/gutenberg_clean_dir.py data/gutenberg/dump/A data/gutenberg/books
    #cat data/gutenberg/books/* | python scripts/clean_sentences.py > data/gutenberg.fr 2> data/gutenberg.fr.log 
    #cat data/gutenberg.fr | sort -u | python scripts/sentencesegmenter.py | sort -u | grep "^[\'\"\«A-Z]" | grep "[\.\,\;\?\!\'\»\"\:]$" > data/gutenberg.fr.txt
}

AFP(){
    cat /nfs/CORPUS/zhangd/AFP/raw/YEAR-MONTH/HEADLINE/HEADLINES_2018_1_fr.txt | python scripts/sentencesegmenter.py | sort -u | grep -v '/' | grep "^[\'\"\«A-Z]" | grep "[\.\,\;\?\!\'\»\"\:]$" | sort -u    > $PWD/data/HEADLINES_2018_1_fr.txt
    cat /nfs/CORPUS/zhangd/AFP/raw/YEAR-MONTH/BODY/BODY_2018_1_fr.txt          | python scripts/sentencesegmenter.py | sort -u | grep -v '/' | grep "^[\'\"\«A-Z]" | grep "[\.\,\;\?\!\'\»\"\:]$" | head -1500 > $PWD/data/BODY_2018_1_fr.txt
    cat /nfs/CORPUS/zhangd/AFP/raw/YEAR-MONTH/BODY/BODY_2017_*_fr.txt          | python scripts/sentencesegmenter.py | sort -u | grep -v '/' | grep "^[\'\"\«A-Z]" | grep "[\.\,\;\?\!\'\»\"\:]$" | sort -u    > $PWD/data/BODY_2017_fr.txt
}

morphalou(){
    python scripts/morphalou31_lexicon.py data/Morphalou3.1_CSV.csv
    python scripts/morphalou31_hphones.py data/Morphalou3.1_CSV.csv
}

vocabs(){
    python vocabs.py -o data/vocabs --one_out_of 50 --lex data/Morphalou3.1_CSV.csv.lexicon.pickle data/BODY_2017_fr.txt data/gutenberg.fr.txt #data/news.2021.fr.txt
    cat data/vocabs.CORREC.freq  | perl -pe 's/^\d+\s+//' | grep -v -E "^[0-9]+$" | grep -v "[[:punct:]][[:punct:]]" | head -50000 > data/vocabs.CORREC.50k
    cat data/vocabs.SHAPES.freq  | perl -pe 's/^\d+\s+//' > data/vocabs.SHAPES
    cat data/vocabs.INLEXI.freq  | perl -pe 's/^\d+\s+//' > data/vocabs.INLEXI
    cat data/Morphalou3.1_CSV.csv.lexicon.lex | cut -f 3 | sort -u > data/vocabs.LFEATS
}

#gutenberg
#news-crawl
#morphalou
#AFP
#vocabs &> log.vocabs &
flex=./data/Morphalou3.1_CSV.csv.lexicon.pickle
fpho=./data/Morphalou3.1_CSV.csv.hphones.pickle
flem=./data/lemrules.cfg
fcor=./data/vocabs.CORREC.50k
flin=./data/vocabs.LFEATS
ferr=./data/vocabs.ERRORS
fsha=./data/vocabs.SHAPES
finl=./data/vocabs.INLEXI

fsrc=./data/HEADLINES_2018_1_fr.txt
#python spacyfy.py --lex $flex -n 8 -o $fsrc.json --shapes $fsha --inlexi $finl $fsrc
#
python noise.py --lex $flex --pho $fpho --lem $flem --errors $ferr --correc $fcor --lfeats $flin --shapes $fsha --inlexi $finl --seed 1234 < $fsrc.json > $fsrc.json.noise

fsrc=./data/BODY_2018_1_fr.txt
#python spacyfy.py --lex $flex -n 8 -o $fsrc.json --shapes $fsha --inlexi $finl $fsrc
#
python noise.py --lex $flex --pho $fpho --lem $flem --errors $ferr --correc $fcor --lfeats $flin --shapes $fsha --inlexi $finl --seed 1234 < $fsrc.json > $fsrc.json.noise

fsrc=./data/news.2007.fr.txt
#python spacyfy.py --lex $flex -n 8 -o $fsrc.json --shapes $fsha --inlexi $finl $fsrc
#
python noise.py --lex $flex --pho $fpho --lem $flem --errors $ferr --correc $fcor --lfeats $flin --shapes $fsha --inlexi $finl --seed 1234 < $fsrc.json > $fsrc.json.noise

fsrc=./data/BODY_2017_fr.txt
#python spacyfy.py --lex $flex -n 32 -o $fsrc.json --shapes $fsha --inlexi $finl $fsrc

fsrc=./data/gutenberg.fr.txt
#python spacyfy.py --lex $flex -n 32 -o $fsrc.json --shapes $fsha --inlexi $finl $fsrc

fsrc=./data/news.2021.fr.txt
#python spacyfy.py --lex $flex -n 32 -o $fsrc.json --shapes $fsha --inlexi $finl $fsrc

#############
### train ###
#############
exp=./exp
mkdir -p $exp

#python train.py --model kkmodel --train data/gutenberg.fr.txt.noise --valid data/HEADLINES_2018_8_fr.txt.noise --err $ferr --cor $fcor --log_file kkmodel.log --cuda &
#python train.py --model kkmodel --train data/gutenberg.fr.txt.noise --valid data/HEADLINES_2018_8_fr.txt.noise --err $ferr --cor $fcor --log_file kkmodel.log --cuda --accum_n_batchs 8 &
#python train.py --model models/kkmodel.00130000.pt --train kk.10k.json --err $ferr --cor $fcor


