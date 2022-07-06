# -*- coding: utf-8 -*-

import sys
import re
import glob
import html
import chardet
import requests
import chardet
import fasttext
import os.path
from tqdm import tqdm

model = fasttext.load_model('/nfs/RESEARCH/senellarta/ressources/lid-systran.bin')
TAGS = re.compile('<[^>]+>')

def remove_tags(s):
    s = html.unescape(s)
    return re.sub(TAGS, '', s);

def get_valid_data(data,fd_log):
    n = len(data)
    nwithin = 0
    Lwithin = []
    L = []
    within = False
    for l in data:
        l = re.sub('\s\s+',' ',l.strip())
        #print("##### {}".format(l))
        if (l.startswith("<p>") or l.startswith("<P>")) and within == False:
            within = True
        if within:
            Lwithin.append(l)
            nwithin += 1
        if (l.endswith("</p>") or l.endswith("</P>")) and within == True:
            within = False
            l = remove_tags(' '.join(Lwithin))
            lid = model.predict(l,k=3)
            if lid[0][0] == '__label__fr' and lid[1][0] >= 0.6:
                L.append(l)
            Lwithin = []
    if len(L):
        fd_log.write('{} {}/{} [{:.2f}%]\t{}\n'.format(len(L),nwithin,n,100.0*nwithin/n,enc))
        fd_log.flush()
    return L
    
idir = sys.argv[1]
odir = sys.argv[2]
os.makedirs(odir, exist_ok=False)
flog = odir+'/log'

fd_log = open(flog, 'w')
files = glob.glob(idir+'/*')
for fin in tqdm(files):
    fd_log.write(fin+'\n')
    base = os.path.basename(fin).replace('.html','.clean')
    fout = odir + '/' + base
    data = open(fin, "rb").read()
    enc = chardet.detect(data)['encoding']
    data = data.decode(enc).encode("utf-8").decode("utf-8") ### input is bin; decode(enc): bin->str(enc); encode(utf-8): str->bin(utf-8); decode(utf-8): bin->str(utf-8)
    data = data.splitlines()
    data = get_valid_data(data,fd_log)
    if len(data):
        with open(fout,'w') as fo:
            for l in data:
                fo.write(l+'\n')
                
close(fd)
