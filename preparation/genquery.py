import random
from collections import Counter

import numpy as np
import os


#generate queries for datasets

Fnms = []
for i in range(20):
    Fnms += ['cast_info/cast_info_' + str(i) + '.txt']

path = 'cast_info/cast_info.meta'
with open(path, 'r', errors='ignore') as f:
    r = f.readline()
    cols = r.split(',')[2].rstrip()

real = []
cate = []

d = len(cols)
for i in range(d):
    if cols[i] == 'C':
        cate += [i]
    if cols[i] == 'R':
        real += [i]

rd = len(real)
cd = len(cate)
print(real)
print(cate)


ff = open('test_query_pair/cast_info/cast_info.txt', 'w')
ff.write('cast_info;0,1,2,3,4,5,6\n')
ol = 0.03


for i in range(100):
    print('query pair ' + str(i))
    pairs = []
    ifcate = []
    for j in range(2):
        pair = []
        cnum = 0
        if cd > 0:
            cnum = random.randint(0, 1)
        rnum = 0
        if cnum == 0:
            rnum = random.randint(2, 4)
            ids = []
            ids = np.random.choice(rd, size=rnum, replace=False, p=None)
            selr = []
            for id in ids:
                selr += [real[id]]
            pair += [selr]
        if cnum == 1:
            ids = []
            ids = np.random.choice(cd, size=cnum, replace=False, p=None)
            selc = [cate[ids[0]]]
            rnum = random.randint(1, 3)
            ids = []
            ids = np.random.choice(rd, size=rnum, replace=False, p=None)
            selr = []
            for id in ids:
                selr += [real[id]]
            pair += [selr]
            pair += [selc]
            ifcate += [len(pairs)]
        pairs += [pair]
    print(pairs[0])
    print(pairs[1])
    rmins = {0: [], 1: []}
    rmaxs = {0: [], 1: []}
    for fnm in Fnms:
        rs = []
        with open(fnm, 'r', errors='ignore') as f:
            rs = f.readlines()
        #fid = fids[fnm]
        n = int(ol * len(rs))
        fid = np.random.choice(len(rs), size=n, replace=False, p=None)
        for id in fid:
            r = rs[id].split('\n')[0].split('|')
            for p in range(len(pairs)):
                if len(rmins[p]) == 0:
                    min = []
                    max = []
                    selr = pairs[p][0]
                    for j in range(len(selr)):
                        d = selr[j]
                        vr = r[d]
                        if vr == '-1':
                            min += [10000000000]
                            max += [-1]
                        else:
                            min += [vr]
                            max += [vr]
                    rmins[p] = min
                    rmaxs[p] = max
                else:
                    min = rmins[p]
                    max = rmaxs[p]
                    selr = pairs[p][0]
                    for j in range(len(selr)):
                        d = selr[j]
                        vr = r[d]
                        if vr == '-1':
                            continue
                        if float(vr) < float(min[j]):
                            min[j] = vr
                        if float(vr) > float(max[j]):
                            max[j] = vr
                    rmins[p] = min
                    rmaxs[p] = max

    if len(ifcate) == 0:
        print('begin print')
        for p in range(len(pairs)):
            selr = pairs[p][0]
            for j in range(len(selr)):
                if j < len(selr)-1:
                    ff.write(str(selr[j]) + ',')
                else:
                    ff.write(str(selr[j]) + ';')
            min = rmins[p]
            for j in range(len(selr)):
                if j < len(selr)-1:
                    ff.write(str(min[j]) + ',')
                else:
                    ff.write(str(min[j]) + ';')
            max = rmaxs[p]
            for j in range(len(selr)):
                if j < len(selr) - 1:
                    ff.write(str(max[j]) + ',')
                else:
                    ff.write(str(max[j]) + ';')
    else:
        pval = {0: [], 1: []}
        for ic in ifcate:
            selc = pairs[ic][1]
            dc = selc[0]
            rds = []
            for fnm in Fnms:
                rs = []
                row = []
                with open(fnm, 'r', errors='ignore') as f:
                    rs = f.readlines()
                    for r in rs:
                        r = r.split('\n')[0].split('|')
                        row += [r[dc]]
                rds += [row]
            flag = True
            print('begin categorical')
            count = 0
            while flag:
                count += 1
                if count > 500:
                    break
                n = np.random.randint(0, len(rds[0]) - 1)
                vc = rds[0][n]
                if vc == 'null':
                    continue
                c = 0
                for row in rds:
                    dnum = (len(row) * ol)
                    if row.count(vc) < dnum:
                        break
                    c += 1
                if c == len(rds):
                    pval[ic] = [vc]
                    flag = False
            if len(pval[ic]) == 0:
                number = Counter(rds[0])
                result = number.most_common()
                for vc in result[0]:
                    if vc != 'null':
                        pval[ic] = [vc]
                        break
        print(pval)
        print('begin print')
        for p in range(len(pairs)):
            if len(pval[p]) > 0:
                selr = pairs[p][0]
                for j in range(len(selr)):
                    ff.write(str(selr[j]) + ',')
                ff.write(str(pairs[p][1][0]) + ';')
                min = rmins[p]
                for j in range(len(selr)):
                    ff.write(str(min[j]) + ',')
                ff.write(str(pval[p][0]) + ';')
                max = rmaxs[p]
                for j in range(len(selr)):
                    ff.write(str(max[j]) + ',')
                ff.write(str(pval[p][0]) + ';')
            else:
                selr = pairs[p][0]
                for j in range(len(selr)):
                    if j < len(selr) - 1:
                        ff.write(str(selr[j]) + ',')
                    else:
                        ff.write(str(selr[j]) + ';')
                min = rmins[p]
                for j in range(len(selr)):
                    if j < len(selr) - 1:
                        ff.write(str(min[j]) + ',')
                    else:
                        ff.write(str(min[j]) + ';')
                max = rmaxs[p]
                for j in range(len(selr)):
                    if j < len(selr) - 1:
                        ff.write(str(max[j]) + ',')
                    else:
                        ff.write(str(max[j]) + ';')
    ff.write('\n')
ff.close()


