import numpy as np
import os

from utils import *

def readd(fname, istest=2, hid=1, hsize=1, sample_rate=1, nqs=50):
    dic = {'Table': [], 'Rows': [], 'Fnms': [], 'Cnts': [], 'Keys': [], 'DimC': [], 'DimR': [], 'Pred_low': [], 'Pred_high': [], 'DVcol': [], 
           'GT_freq': [], 'GT_dv': [], 'Types': [], 'Samples': [], 'lb': [], 'ub': []}
    path = os.path.dirname(fname)
    lid = 0
    with open(fname, 'r') as f:
        lines = f.readlines()
        nln = int(lines[0])
        for l in lines[1:]:
            if (l.startswith("#")):
                continue
            lid += 1

            d = l.split(';')
            dR = list(map(int, filter(None, d[1].split(","))))
            dC = list(map(int, filter(None, d[2].split(","))))
            dic['Table'] += [d[0]] # path of meta file
            dic['DimR'] += [dR] # the number of cols
            dic['DimC'] += [dC] 

            if (len(d) >= 4):
                dvcols = []
                predl = []
                predh = []
                for n in range(nqs):
                    dvcols += [list(map(int, filter(None, d[3 + 2 * n].split(','))))] # col id (attribute) of each query
                    pred = list(filter(None, d[4 + 2 * n].split(','))) # range of attribute
                    lp = int(len(pred) / 2)
                    predl += [pred[:lp]] # the first half is lower bound
                    predh += [pred[lp:]] # the second half is upper bound
                dic['DVcol'] += [dvcols] # attributes of queries
                dic['Pred_low'] += [predl]
                dic['Pred_high'] += [predh]

            with open(path + '/' + d[0], 'r') as fmeta:
                mline = fmeta.readline().split(',')
                cols = int(mline[0]) 
                total_dv = int(mline[1])
                dic['Types'] += [mline[2].rstrip()] # type of attributes (real or categorical)
                keys, cnts = [], []
                for i in range(cols):
                    k, m = [], []
                    fmeta.readline()
                    cur = int(0)
                    while (True):
                        line = fmeta.readline().split(',')
                        p = int(line[-1]) 
                        k += [",".join(line[:-1])] #line[0]-->distinct values in an attribute
                        m += [int(p) - cur] #the number of a distinct value
                        cur = int(p)
                        if (p == total_dv + 1):
                            break;
                    cnts += [m] if (i in dR or i in dC or nqs == -1) else [[]]
                    keys += [k] if (i in dR or i in dC or nqs == -1) else [[]]
                line = fmeta.readline().split(';')
                fns = []
                fns += [path + '/' + os.path.dirname(d[0]) + '/' + line[0]]
                dic['Fnms'] += [fns]  #the path of data
                dic['Cnts'] += [cnts]
                dic['Keys'] += [keys]
                dic['Samples'] += [path + '/' + os.path.dirname(d[0]) + '/sample/' + line[0]]  # sample file
    print('Loading ' + fname + ' with ' + str(len(dic['Fnms'])) + ' sets.')
    return dic

def read_raw(icache, fnames, ds):
    rows = []
    for fname in fnames:
        if fname not in icache:
            with open(fname, 'r', errors='ignore') as f:
                icache[fname] = [r for r in np.loadtxt(f, delimiter='|', dtype=str)]
        rows += [r[ds] for r in icache[fname]]
    return np.array(rows)

def bkt_raw(input, keys, nss):
    id = max(0, min(len(keys) - 1, np.searchsorted(keys, input, side='right') - 1))
    return int(np.minimum(nss - 1, (id * nss) / len(keys)))


def readr(icache, bcache, fnames, types, nsssort, ds, cnts=[]):
    VV = [[] for _ in range(len(ds))]
    for d in range(len(ds)):
        for i in range(len(fnames)):
            ckey = fnames[i] + ';' + str(ds[d]) + ';' + str(nsssort[d])
            if ckey not in bcache:
                raw = read_raw(icache, [fnames[i]], ds[d])
                if types[d] == "R":
                    raw = raw.astype(float)
                elif types[d] == "D":
                    raw = [parse_date(r) for r in raw]
                    raw = np.array([(date(r[0], r[1], r[2]) - date(1900, 1, 1)).days for r in raw])
                bcache[ckey] = raw
            VV[d] += [bcache[ckey]]
        VV[d] = np.hstack(VV[d])
    return VV


def reads(fnames, types, ds, lens):
    bcache = {}
    VV = []
    for d in range(len(ds)):
        for i in range(len(fnames)):
            ckey = fnames[i] + ';' + str(ds[d])
            if ckey not in bcache:
                raw = read_raw({}, [fnames[i]], ds[d])[:int(lens)]
                if types[d] == "R":
                    raw = raw.astype(float)
                elif types[d] == "D":
                    raw = [parse_date(r) for r in raw]
                    raw = [(date(r[0], r[1], r[2]) - date(1900, 1, 1)).days for r in raw]
                else:
                    pass
                bcache[ckey] = raw
            VV += [bcache[ckey]]
    return VV


def parse_raw(VV, keys, nsssort):
    nd = len(VV)
    nr = len(VV[0])
    v = np.zeros((nr, nd)).astype(int)
    for d in range(nd):
        ids = np.maximum(0, np.minimum(len(keys[d]) - 1, np.searchsorted(keys[d], VV[d], side='right') - 1))
        v[:, d] = np.minimum(nsssort[d] - 1, ids * nsssort[d] / len(keys[d])).astype(int)
    return v

def readp(pred_low, pred_high, keys, nsssort):
    rect = 1.0
    pred_low_s, pred_high_s = [], []
    for d in range(len(pred_low)):
        pred_low_s += [bkt_raw(pred_low[d], keys[d], nsssort[d])]
        pred_high_s += [bkt_raw(pred_high[d], keys[d], nsssort[d])]
        rect *= (pred_high_s[d] >= pred_low_s[d])
    return pred_low_s, pred_high_s, rect

def read_feat(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
        feat = []
        meta = list(map(float, lines[0].split(',')))
        for i in range(len(meta)):
            feat += [list(map(float, lines[i+1]).split(','))]
    return meta, feat

def write_feat(fname, feat, meta):
    with open(fname, 'w') as f:
        f.write(','.join(meta))
        for i in range(len(meta)):
            f.write(','.join(feat[i]))