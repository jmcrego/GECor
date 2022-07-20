import sys
import json

class Conll():
    def __init__(self, tags=False):
        self.tags = tags
    
    def __call__(self,line):
        out = []
        for w in line:
            wout = []
            if 'raw' in w:
                tag = 'raw:' if self.tags else ''
                wout.append(tag+w['raw'])
            if 'iraw' in w:
                tag = 'iraw:' if self.tags else ''
                wout.append(tag+"{}".format('-'.join(list(map(str,w['iraw'])))))
            if 'shp' in w:
                tag = 'shp:' if self.tags else ''
                wout.append(tag+w['shp'])
            if 'ishp' in w:
                tag = 'ishp:' if self.tags else ''
                wout.append(tag+str(w['ishp']))
            if 'lex' in w:
                tag = 'lex:' if self.tags else ''
                wout.append(tag+str(w['lex']))
            if 'ilex' in w:
                tag = 'ilex:' if self.tags else ''
                wout.append(tag+str(w['ilex']))
            if 'err' in w:
                tag = 'err:' if self.tags else ''
                wout.append(tag+w['err'])
            if 'ierr' in w:
                tag = 'ierr:' if self.tags else ''
                wout.append(tag+str(w['ierr']))
            if 'cor' in w:
                tag = 'cor:' if self.tags else ''
                wout.append(tag+w['cor'])
            if 'icor' in w:
                tag = 'icor:' if self.tags else ''
                wout.append(tag+str(w['icor']))
            if 'iCOR' in w:
                tag = 'iCOR:' if self.tags else ''
                wout.append(tag+"{}".format('-'.join(list(map(str,w['iCOR'])))))
            if 'lng' in w:
                tag = 'lng:' if self.tags else ''
                wout.append(tag+w['lng'])
            if 'ilng' in w:
                tag = 'ilng:' if self.tags else ''
                wout.append(tag+str(w['ilng']))
            if 'plm' in w:
                tag = 'plm:' if self.tags else ''
                wout.append(tag+w['plm'])
            out.append('\t'.join(wout))
        return '\n'.join(out) + '\n'
    
if __name__ == '__main__':

    conll = Conll(tags=True)
    for l in sys.stdin:
        print(conll(json.loads(l)))

    
