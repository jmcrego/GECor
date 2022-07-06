import sys
import json

class Conll():
    def __init__(self):
        pass
    
    def __call__(self,line):
        out = []
        for w in line:
            wout = []
            if 'r' in w:
                wout.append(w['r'])
            if 'i' in w:
                wout.append("{}".format('-'.join(list(map(str,w['i'])))))
            if 's' in w:
                wout.append(w['s'])
            if 'is' in w:
                wout.append(str(w['is']))
            if 't' in w:
                wout.append(str(w['t']!=''))
            if 'E' in w:
                wout.append(w['E'])
            if 'iE' in w:
                wout.append(str(w['iE']))
            if 'C' in w:
                wout.append(w['C'])
            if 'iC' in w:
                wout.append(str(w['iC']))
            if 'iCC' in w:
                wout.append("{}".format('-'.join(list(map(str,w['iCC'])))))
            if 'L' in w:
                wout.append(w['L'])
            if 'iL' in w:
                wout.append(str(w['iL']))
            if 'plm' in w:
                wout.append(w['plm'])
            out.append('\t'.join(wout))
        return '\n'.join(out) + '\n'
    
if __name__ == '__main__':

    conll = Conll()
    for l in sys.stdin:
        print(conll(json.loads(l)))

    
