import random
import csv
import numpy as np
import os

inf = -1



path = 'dataset_public/test'
with open(path + '/cast_info/cast_info.meta', 'r') as fmeta:
    allline = fmeta.readlines()
    line = allline[len(allline) - 1].split(';')
    fns = []
    for i in range(int(line[1])):
        fns += [path + '/cast_info/' + line[0].replace("\n", "").replace("$n$", str(i))]
rows = []
for fname in fns:
    print(fname)
    with open(fname, 'r', errors='ignore') as f:
        lines = f.readlines()
        rows += [r for r in lines]
f = open('cast_info/cast_info.txt', 'w')
rand = np.random.randint(500, 3000)
sr = float(rand) / float(10000)
N = range(len(rows))
m = int(len(N) * sr)
ids = np.random.choice(N, size=m, replace=False, p=None)
print(ids)
for j in range(len(ids)):
    r = rows[ids[j]]
    f.write(r)
f.close()

path = 'cast_info/cast_info.txt'
allRows = []
with open(path, 'r', errors='ignore') as f:
    allRows = f.readlines()
N = len(allRows)
print(N)

ids = np.random.choice(N, size=5000, replace=False, p=None)
ff = open('sample/cast_info.txt', 'w')
for j in range(len(ids)):
    r = allRows[ids[j]]
    ff.write(r)
f.close()


