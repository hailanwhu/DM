import random
import copy
import datetime
import time
from collections import OrderedDict

def exeQueries(rs, dvcols, predls, predhs, types):
    Q_d = []
    starttime = time.time()
    for i in range(len(dvcols)):
        dvcol = dvcols[i]
        predl = predls[i]
        predh = predhs[i]
        q_d = []
        for row in rs:
            r = row.split('\n')[0].split('|')
            flag = True
            for j in range(len(dvcol)):
                num = dvcol[j]
                if types[num] == 'R':
                    low = float(predl[j])
                    high = float(predh[j])
                    value = float(r[num])
                    if value < low or value > high:
                        flag = False
                        #break
                else:
                    low = predl[j]
                    high = predh[j]
                    value = r[num]
                    if value != low or value != high:
                        flag = False
                        #break
            if flag:
                q_d += [row]
        Q_d = Q_d + q_d
        Q_d = list(OrderedDict.fromkeys(Q_d))
    endtime = time.time()
    cardtime = float(endtime - starttime)
    return Q_d, cardtime

def mergeRows(R_max, R):
    starttime = time.time()
    newR = R_max + R
    newR = list(OrderedDict.fromkeys(newR))
    endtime = time.time()
    mergetime = float(endtime - starttime)
    return newR, mergetime


def baseline(Fnms, B, price, rows, dvcols, predls, predhs, types):
    ff = open('baseline.txt', 'w')

    U = 0
    d = ''
    p = 0
    total_cardtime = 0
    total_mergetime = 0
    starttime = time.time()
    for fnm in Fnms:
        p_d = price[fnm]
        rs = rows[fnm]
        Q_d, cardtime = exeQueries(rs, dvcols, predls, predhs, types)
        total_cardtime = total_cardtime + cardtime
        if len(Q_d) > U and p_d <= B:
            U = len(Q_d)
            d = fnm
            p = p_d
    print(d)
    print(U)

    S = []
    R_max = []
    R = {}
    for fnm in Fnms:
        R[fnm] = []
    P = 0
    u_max = 0
    while len(Fnms) > 0 and P < B:
        d_max = ''
        g_max = 0
        p_max = 0
        newR = []
        for fnm in Fnms:
            #print(fnm)
            p_d = price[fnm]
            rs = rows[fnm]
            if len(R[fnm]) == 0:
                Q_d, cardtime = exeQueries(rs, dvcols, predls, predhs, types)
                total_cardtime = total_cardtime + cardtime
                R[fnm] = Q_d
            #print(len(R[fnm]))
            RS, mergetime = mergeRows(R_max, R[fnm])
            total_mergetime = total_mergetime + mergetime
            u = len(RS)
            #print(u)
            g = float(u-u_max) / float(p_d)
            if g > g_max and P + p_d <= B:
                g_max = g
                d_max = fnm
                p_max = p_d
                newR = RS
        if d_max != '':
            S += [d_max]
            Fnms.remove(d_max)
            P += p_max
            R_max = newR
            u_max = len(R_max)
            print(d_max)
            print(u_max)
            print(P)
        else:
            break
    endtime = time.time()
    totaltime = float(endtime - starttime)

    if u_max < U:
        print('found dataset:')
        print(fnm)
        print('utility: ' + str(U))
        print('total time: ' + str(totaltime))
        print('merge time: ' + str(total_mergetime))
        print('card time: ' + str(total_cardtime))
        sf = fnm.split('_')
        d = sf[len(sf) - 1].split('.')[0]
        ff.write('found dataset: ' + str(d) + '\n')
        ff.write('utility: ' + str(U) + '\n')
        ff.write('total time: ' + str(totaltime) + '\n')
        ff.write('merge time: ' + str(total_mergetime) + '\n')
        ff.write('card time: ' + str(total_cardtime) + '\n')
        ff.write('budget: ' + str(B) + '\n')
        ff.write('used budget: ' + str(p) + '\n')
    else:
        print('found dataset:')
        ff.write('found dataset: ')
        for d in S:
            print(d + ', ', end='')
            sf = d.split('_')
            ds = sf[len(sf) - 1].split('.')[0]
            ff.write(str(ds) + ', ')
        print()
        ff.write('\n')
        print('utility: ' + str(u_max))
        print('total time: ' + str(totaltime))
        print('merge time: ' + str(total_mergetime))
        print('card time: ' + str(total_cardtime))
        ff.write('utility: ' + str(u_max) + '\n')
        ff.write('total time: ' + str(totaltime) + '\n')
        ff.write('merge time: ' + str(total_mergetime) + '\n')
        ff.write('card time: ' + str(total_cardtime) + '\n')
        ff.write('budget: ' + str(B) + '\n')
        ff.write('used budget: ' + str(P) + '\n')
    ff.close()


if __name__ == '__main__':
    Fnms = []
    for i in range(20):
        Fnms += ['data/airline/airline_' + str(i) + '.txt']

    with open('data/airline/airline_0.meta', 'r') as fmeta:
        line = fmeta.readline()
    types = line.split(',')[2].split('\n')[0]

    price = {}
    rows = {}
    B = 0
    with open('weight.txt', 'r', errors='ignore') as f:
        weight = f.readlines()
    i = 0
    for fnm in Fnms:
        with open(fnm, 'r', errors='ignore') as f:
            rs = f.readlines()
        rows[fnm] = rs
        w = float(weight[i].split('\n')[0])
        price[fnm] = int(len(rs) * w)
        B += price[fnm]
        i += 1
    B = B * 0.5
    print('Budget: ' + str(B))

    path = 'full_airline.txt'
    dvcols = []
    predls = []
    predhs = []
    with open(path, 'r', errors='ignore') as f:
        lines = f.readlines()
        l = lines[1]
        d = l.split(';')
        for n in range(20):
            dvcols += [list(map(int, filter(None, d[3 + 2 * n].split(','))))]  # col id (attribute) of each query
            pred = list(filter(None, d[4 + 2 * n].split(',')))  # range of attribute
            lp = int(len(pred) / 2)
            predls += [pred[:lp]]  # the first half is lower bound
            predhs += [pred[lp:]]  # the second half is upper bound

    baseline(Fnms, B, price, rows, dvcols, predls, predhs, types)