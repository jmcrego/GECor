# -*- coding: utf-8 -*-

import sys
import re
import html
import chardet
import requests
import chardet
import fasttext

def avgletters(s):
    if len(s) == 0:
        return 0.0
    n_alpha = 0
    n_other = 0
    for c in s:
        if c.isalpha():
            n_alpha += 1
        else:
            n_other += 1
    avg = 1.0 * n_alpha / (n_other+n_alpha)
    if avg <  0.5:
        sys.stderr.write('avgletters error {:.1f}\t{}\n'.format(avg,l))
        return False
    return True

def balanced_op_cl(l):
    op = l.count('«')
    cl = l.count('»')
    if op != cl:
        sys.stderr.write("unbalanced(op+cl)\t{}\n".format(l))
        return False
    return True

def my_replace_wd(m):
    sys.stderr.write('matched_wd: {}{} => {}\t{}\n'.format(m.group(1),m.group(2),m.group(1),l))
    return m.group(1)

def my_replace_initial_d(m):
    sys.stderr.write('matched_initial_d: {}{} => {}\t{}\n'.format(m.group(1),m.group(2),m.group(2),l))
    return m.group(2)

for l in sys.stdin:
    l = re.sub('\s\s+',' ',l.strip())

    if not balanced_op_cl(l):
        continue

    if re.search(r'^\d',l):
        sys.stderr.write('sentence with numbers0\t{}\n'.format(l))
        continue

    if re.search(r'[^ \d]\d',l):
        sys.stderr.write('sentence with numbers1\t{}\n'.format(l))
        continue
    
    if re.search(r'\d[^ \d]',l):
        sys.stderr.write('sentence with numbers2\t{}\n'.format(l))
        continue
    
#    l2 = re.sub("^(\[*\d+\]*\s*)([A-Z])", my_replace_initial_d, l)
#    while l2 != l:
#        l = l2
#        l2 = re.sub("^(\[*\d+\]*\s*)([A-Z])", my_replace_initial_d, l)
        
#    l2 = re.sub('([^\d \[]+)(\[*\d+\]*)', my_replace_wd, l, re.UNICODE)
#    while l2 != l:
#        l = l2
#        l2 = re.sub('([^\d \[]+)(\[*\d+\]*)\s', my_replace_wd, l, re.UNICODE)

    if l.endswith('—') and len(l) > 1:
        sys.stderr.write('deleted ending —\t{}\n'.format(l))
        l = l[:-1]

    if l.startswith('—') and len(l) > 1:
        sys.stderr.write('deleted starting —\t{}\n'.format(l))
        l = l[1:]

    l = re.sub("([^ ])(--)",r"\1 \2",l, re.UNICODE)
    l = re.sub("([^ ])(—)",r"\1 \2",l, re.UNICODE)
        
    l = l.strip()
    
    if len(l) < 4:
        sys.stderr.write('len={} error\t{}\n'.format(len(l),l))
        continue

    if not avgletters(l):
        continue

    
    sys.stderr.write('ok\n')
    print(l)

    
