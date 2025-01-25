import warnings

warnings.filterwarnings('ignore')

from parameters import *
from build_iris import *
from build_sparse_hist import *
from preproc_card import *
import copy
import datetime
import time

os.environ["CUDA_DEVICE_ORDER"] = "1"

def cardinality_first(E, Q_p, options, emb, model_card, model_emb):
    feats = E['feat']
    chs = E['ch']  # ch[0]: column set ch[3]: model


    data_emb = []
    starttime1 = time.time()
    qc = {} # query - colset + feat
    cq = {} # colset - query embedding
    for qi in Q_p:
        dvcol = Q_p[qi][0]
        pred_low_r = Q_p[qi][1]
        pred_high_r = Q_p[qi][2]

        chsn = []
        dvs = dvcol.copy()
        for j in range(len(chs)):
            if len(chs[j][0]) > 1 and chs[j][3] != 'AVI' and set(chs[j][0]).issubset(set(dvs)):
                chsn += [[chs[j], feats[j]]]

        # find colsets
        colsets = {}
        cf = {}
        while len(chsn) > 0:
            colsets[';'.join(map(str, chsn[0][0]))] = chsn[0][1]
            cf[str(chsn[0][0][-1])] = chsn[0][1]
            clique = chsn[0][0][0].copy()
            chsn0 = []
            for j in range(len(chsn) - 1):
                if len(set(chsn[j + 1][0][0]).intersection(set(chsn[0][0][0]))) == 0 and len(
                        set(chsn[j + 1][0][0]).intersection(set(clique))) == 0:
                    chsn0 += [chsn[j + 1]]
            chsn = chsn0
        qc[qi] = cf

        c = 0
        pred_emb = []
        sketch = []
        pad_ske = []
        pad_pred = []
        for colset, feat in colsets.items():
            ns = [int(colset.split(';')[-2].split(',')[0].split('[')[1]), int(colset.split(';')[-2].split(',')[1].split(']')[0])]
            cols = [int(colset.split(';')[-1].split(',')[0].split('[')[1]), int(colset.split(';')[-1].split(',')[1].split(']')[0])]
            ky, pl, ph = [], [], []
            for d in range(len(cols)):
                pl += [pred_low_r[dvs.index(cols[d])]]
                ph += [pred_high_r[dvs.index(cols[d])]]
                ky += [KEYo[ns[d]][cols[d]]]
            PLm, PHm, _ = readp(pl, ph, ky, ns)

            # Compute Iris predictions
            for d in range(len(cols)):
                PLm[d] = int(round(PLm[d] * (options.neb - 1) / (ns[d] - 1)))
                PHm[d] = int(round(PHm[d] * (options.neb - 1) / (ns[d] - 1)))
            PLm, PHm = np.maximum(PLm, -1) + 1, np.maximum(PHm, -1) + 1
            pemb = np.hstack([emb[str(PLm[0]) + ',' + str(PLm[1])], emb[str(PHm[0]) + ',' + str(PHm[1])]])
            cq[str(cols)] = pemb

            if c == 0:
                sketch = np.array(feat).reshape(1, 1, options.nr)
                pred_emb = np.array(pemb).reshape(1, 1, 2 * options.nr)
                pad_ske = sketch
                pad_pred = pred_emb
            else:
                pred_emb = np.concatenate((pred_emb, np.array(pemb).reshape(1, 1, 2 * options.nr)), axis=2)
                sketch = np.concatenate((sketch, np.array(feat).reshape(1, 1, options.nr)), axis=2)
            c += 1
        while sketch.shape[2] < options.nr * options.nx:
            # pad_ske = np.zeros((1, 1, options.nr * options.nx - sketch.shape[2]))
            sketch = np.concatenate((sketch, pad_ske), axis=2)
        while pred_emb.shape[2] < 2 * options.nr * options.nx:
            # pad_pred = np.zeros((1, 1, 2 * options.nr * options.nx - pred_emb.shape[2]))
            pred_emb = np.concatenate((pred_emb, pad_pred), axis=2)
        data_emb += [[sketch, pred_emb]]
    endtime1 = time.time()
    another = float(endtime1 - starttime1)

    embtime = 0
    vd = np.concatenate(([data_emb[0][0], data_emb[1][0]]), axis=2)
    vq = np.concatenate(([data_emb[0][1], data_emb[1][1]]), axis=2)
    starttime = time.time()
    demb = model_emb.predict([vd, vq])
    endtime = time.time()
    embtime = embtime + float(endtime - starttime)
    vd = demb[0]
    vq = demb[1]

    qi = 2
    while qi < len(Q_p):
        vd = np.concatenate(([vd, data_emb[qi][0]]), axis=2)
        vq = np.concatenate(([vd, data_emb[qi][1]]), axis=2)
        starttime = time.time()
        demb = model_emb.predict(
            [vd.reshape(1, 1, 2 * options.nr * options.nx), vq.reshape(1, 1, 4 * options.nr * options.nx)])
        endtime = time.time()
        embtime = embtime + float(endtime - starttime)
        vd = demb[0]
        vq = demb[1]
        qi += 1

    starttime = time.time()
    card = model_card.predict([vd.reshape(1, 1, options.nr * options.nx), vq.reshape(1, 1, 2 * options.nr * options.nx)])[0][
        0][0]
    endtime = time.time()
    cardtime = float(endtime - starttime)

    return card, qc, cq, embtime, cardtime, another

def cardinality(qc, cq, options, model_card, model_emb):

    data_emb = []
    starttime1 = time.time()
    for qi, colsets in qc.items():
        c = 0
        pred_emb = []
        sketch = []
        pad_ske = []
        pad_pred = []
        for colset, feat in colsets.items():
            cols = colset
            pemb = cq[cols]
            if c == 0:
                sketch = np.array(feat).reshape(1, 1, options.nr)
                pred_emb = np.array(pemb).reshape(1, 1, 2 * options.nr)
                pad_ske = sketch
                pad_pred = pred_emb
            else:
                pred_emb = np.concatenate((pred_emb, np.array(pemb).reshape(1, 1, 2 * options.nr)), axis=2)
                sketch = np.concatenate((sketch, np.array(feat).reshape(1, 1, options.nr)), axis=2)
            c += 1

        while sketch.shape[2] < options.nr * options.nx:
            sketch = np.concatenate((sketch, pad_ske), axis=2)
        while pred_emb.shape[2] < 2 * options.nr * options.nx:
            pred_emb = np.concatenate((pred_emb, pad_pred), axis=2)
        data_emb += [[sketch, pred_emb]]

    endtime1 = time.time()
    another = float(endtime1 - starttime1)

    embtime = 0
    vd = np.concatenate(([data_emb[0][0], data_emb[1][0]]), axis=2)
    vq = np.concatenate(([data_emb[0][1], data_emb[1][1]]), axis=2)
    starttime = time.time()
    demb = model_emb.predict([vd, vq])
    endtime = time.time()
    embtime = embtime + float(endtime - starttime)
    vd = demb[0]
    vq = demb[1]

    qi = 2
    while qi < len(qc):
        vd = np.concatenate(([vd, data_emb[qi][0]]), axis=2)
        vq = np.concatenate(([vd, data_emb[qi][1]]), axis=2)
        starttime = time.time()
        demb = model_emb.predict(
            [vd.reshape(1, 1, 2 * options.nr * options.nx), vq.reshape(1, 1, 4 * options.nr * options.nx)])
        endtime = time.time()
        embtime = embtime + float(endtime - starttime)
        vd = demb[0]
        vq = demb[1]
        qi += 1

    starttime = time.time()
    card = model_card.predict([vd.reshape(1, 1, options.nr * options.nx), vq.reshape(1, 1, 2 * options.nr * options.nx)])[0][
        0][0]
    endtime = time.time()
    cardtime = float(endtime - starttime)

    return card, embtime, cardtime, another


def mergeEmbedding(qc, QC_max, model_merge):
    QC = {}
    starttime = time.time()
    for qi, colsets in QC_max.items():
        newcolsets = {}
        for colset, feat in colsets.items():
            if colset not in qc[qi]:
                newcolsets[colset] = feat
            else:
                feat1 = qc[qi][colset]
                f = np.concatenate((feat.reshape(1, 1, options.nr), feat1.reshape(1, 1, options.nr)), axis=2)
                newfeat = model_merge.predict(f)[0][0]
                newcolsets[colset] = newfeat
        QC[qi] = newcolsets

    endtime = time.time()
    mergetime = float(endtime - starttime)

    return QC, mergetime


def um(Es, weights, weights1, options, emb, Fnms, B):
    ff = open('online.txt', 'w')

    model_card = get_query_model(options.nr, options.nx, options.nfc, options.nn)
    for i in range(len(model_card.layers)):
        if isinstance(model_card.layers[i], keras.layers.Dense):
            model_card.layers[i].set_weights(weights[model_card.layers[i].name])

    model_emb = get_data_emb_model(options.nr, options.nx, options.nfc, options.nn)
    for i in range(len(model_emb.layers)):
        if isinstance(model_emb.layers[i], keras.layers.Dense):
            model_emb.layers[i].set_weights(weights[model_emb.layers[i].name])

    model_merge = get_merge_emb_model(options.nr, options.nfc, options.nn)
    for i in range(len(model_merge.layers)):
        if isinstance(model_merge.layers[i], keras.layers.Dense):
            model_merge.layers[i].set_weights(weights1[model_merge.layers[i].name])

    tid = 0
    Q = np.arange(len(dic['DVcol'][tid]))
    Q_p = {}
    for i in range(len(Q)):
        qi = Q[i]
        PL = dic['Pred_low'][tid][qi]
        PH = dic['Pred_high'][tid][qi]
        dvcol = dic['DVcol'][tid][qi]
        pred_low_r = PL
        pred_high_r = PH
        for d in range(len(PL)):
            if (TYPEs[dvcol[d]] == 'R'):
                pred_low_r[d] = float(pred_low_r[d])
                pred_high_r[d] = float(pred_high_r[d])
            elif (TYPEs[dvcol[d]] == 'D'):
                dl, dh = parse_date(pred_low_r[d]), parse_date(pred_high_r[d])
                pred_low_r[d] = (date(dl[0], dl[1], dl[2]) - date(1900, 1, 1)).days
                pred_high_r[d] = (date(dh[0], dh[1], dh[2]) - date(1900, 1, 1)).days
        Q_p[qi] = [dvcol, pred_low_r, pred_high_r]

    U = 0
    d = ''
    price = 0
    total_embtime = 0
    total_cardtime = 0
    total_another = 0
    total_mergetime = 0
    dqc = {} #dataset - query - colset + feat
    dcq = {} # dataset - colset - query embedding
    for fnm in Fnms:
        E_d = Es[fnm]
        p_d = E_d['price'][0]
        nrows = E_d['rows'][0]
        card, qc, cq, embtime, cardtime, another = cardinality_first(E_d, Q_p, options, emb, model_card, model_emb)
        if card > U and p_d <= B:
            U = int(card * nrows)
            d = fnm
            price = p_d
        total_embtime = total_embtime + embtime
        total_cardtime = total_cardtime + cardtime
        total_another = total_another + another
        dqc[fnm] = qc
        dcq[fnm] = cq
    print(d)
    print(U)

    S = []
    QC_max = {}
    CQ_max = {}
    QC = {}
    P = 0
    u_max = 0
    n = 0
    while len(Fnms) > 0 and P < B:
        d_max = ''
        g_max = 0
        p_max = 0
        n_max = 0
        for fnm in Fnms:
            E_d = Es[fnm]
            p_d = E_d['price'][0]
            nrows = E_d['rows'][0]
            qc = dqc[fnm]
            cq = dcq[fnm]
            if len(S) == 0:
                card, embtime, cardtime, another = cardinality(qc, cq, options, model_card, model_emb)
                u = int(card * nrows)
                g = float(u) / float(p_d)
                if g > g_max and p_d <= B:
                    g_max = g
                    d_max = fnm
                    CQ_max = cq
                    QC = qc
                    p_max = p_d
                    n_max = nrows
                total_embtime = total_embtime + embtime
                total_cardtime = total_cardtime + cardtime
                total_another = total_another + another
            else:
                QC_mg, mergetime = mergeEmbedding(qc, QC_max, model_merge)
                card, embtime, cardtime, another = cardinality(QC_mg, CQ_max, options, model_card, model_emb)
                u = int(card * (n + nrows))
                g = float(u - u_max) / float(p_d)
                if g > g_max and P + p_d <= B:
                    g_max = g
                    d_max = fnm
                    QC = QC_mg
                    p_max = p_d
                    n_max = nrows
                total_embtime = total_embtime + embtime
                total_cardtime = total_cardtime + cardtime
                total_another = total_another + another
                total_mergetime = total_mergetime + mergetime
        if d_max != '':
            QC_max = QC
            S += [d_max]
            Fnms.remove(d_max)
            P += p_max
            u_max = u_max + g_max * p_max
            n = n + n_max
            print(d_max)
            print(u_max)
            print(P)
            print('merge time: ' + str(total_mergetime))
            print('emb time: ' + str(total_embtime))
            print('card time: ' + str(total_cardtime))
        else:
            break


    if u_max < U:
        print('found dataset:')
        print(fnm)
        print('utility: ' + str(U))
        totaltime = total_embtime + total_cardtime + total_another + total_mergetime
        print('total time: ' + str(totaltime))
        print('merge time: ' + str(total_mergetime))
        print('emb time: ' + str(total_embtime))
        print('card time: ' + str(total_cardtime))
        sf = fnm.split('_')
        d = sf[len(sf)-1].split('.')[0]
        ff.write('found dataset: ' + str(d) + '\n')
        ff.write('utility: ' + str(U) + '\n')
        ff.write('total time: ' + str(totaltime) + '\n')
        ff.write('merge time: ' + str(total_mergetime) + '\n')
        ff.write('emb time: ' + str(total_embtime) + '\n')
        ff.write('card time: ' + str(total_cardtime) + '\n')
        ff.write('budget: ' + str(B) + '\n')
        ff.write('used budget: ' + str(price) + '\n')
    else:
        print('found dataset:')
        for d in S:
            print(d + ', ', end='')
            sf = d.split('_')
            ds = sf[len(sf)-1].split('.')[0]
            ff.write(str(ds) + ', ')
        print()
        ff.write('\n')
        print('utility: ' + str(u_max))
        totaltime = total_embtime + total_cardtime + total_another + total_mergetime
        print('total time: ' + str(totaltime))
        print('merge time: ' + str(total_mergetime))
        print('emb time: ' + str(total_embtime))
        print('card time: ' + str(total_cardtime))
        ff.write('utility: ' + str(u_max) + '\n')
        ff.write('total time: ' + str(totaltime) + '\n')
        ff.write('merge time: ' + str(total_mergetime) + '\n')
        ff.write('emb time: ' + str(total_embtime) + '\n')
        ff.write('card time: ' + str(total_cardtime) + '\n')
        ff.write('budget: ' + str(B) + '\n')
        ff.write('used budget: ' + str(P) + '\n')

    ff.close()


if __name__ == '__main__':
    options = parse_arg()

    dic = readd(options.data_dir + '/' + options.input_fnm, sample_rate=1, nqs=20)
    options.max_atom_budget *= options.storage
    if options.storage < 0.5:
        options.sample_size *= 0.5
    print('Storage budget ' + str(options.storage * 4 * len(dic['Types'][0])) + 'KB, max atom budget ' + str(
        4 * options.max_atom_budget / 1024) + 'KB, sample size ' + str(options.sample_size) + ' rows')
    options.storage = len(dic['Types'][0]) * (options.storage * 4 - options.neb / 1024 - options.sample_size / 256)
    print('Storage budget exlucding overhead ' + str(options.storage) + 'KB')

    print("Extracting embedding weights from pre-trained model (can be cached offline)..")
    weights = {}
    model_path = options.model_fnm + '/card/model_lrelu'
    iris_full_model = load_model(model_path).layers[-2]
    weights['Embedding'] = iris_full_model.layers[3].get_weights()
    for i in range(len(iris_full_model.layers)):
        if isinstance(iris_full_model.layers[i], keras.layers.Dense):
            weights[iris_full_model.layers[i].name] = iris_full_model.layers[i].get_weights()
            
    weights1 = {}
    model_path1 = options.model_fnm + '/emb/model_lrelu'
    merge_model = load_model(model_path1).layers[-2]
    for i in range(len(merge_model.layers)):
        if isinstance(merge_model.layers[i], keras.layers.Dense):
            weights1[merge_model.layers[i].name] = merge_model.layers[i].get_weights()

    if options.extract_emb:
        extract_emb(iris_full_model, model_path, options.neb, options.ncd)

    Fnms = []
    for i in range(len(dic['Fnms'])):
        Fnms += [dic['Fnms'][i][0]]

    emb = pickle.load(open('tmp/emb-model_lrelu.pkl', 'rb'))
    Es = {}
    B = 0
    with open('weight.txt', 'r',
              errors='ignore') as f:
        weight = f.readlines()
    j = 0
    for fnm in Fnms:
        E = {}
        ff = pickle.load(
            open('tmp/feature-' + os.path.splitext(os.path.basename(fnm))[0] + '-' + str(options.storage) + '.pkl',
                 'rb'))
        fb = pickle.load(open(
            'tmp/bucket-' + os.path.splitext(os.path.basename(fnm))[0] + '-' + str(options.storage) + '.pkl' if len(
                options.nusebucket) == 0 else options.nusebucket, 'rb'))

        feats = ff['feat']
        chs = ff['ch']
        sample = ff['sample']
        TYPEs = ff['TYPEs']
        KEYo = fb['KEYo']
        for i in range(len(feats)):
            if chs[i][3] == 'Hist' or (chs[i][3] == 'AVI' and len(chs[i][0]) == 1):
                feats[i] = warm_up_hist(feats[i], len(chs[i][0]))

        E['feat'] = feats
        E['ch'] = chs
        E['sample'] = sample
        E['TYPEs'] = TYPEs
        E['KEYo'] = KEYo

        # computing prices of datasets
        with open(fnm, 'r', errors='ignore') as f:
            rs = f.readlines()
        # w = float(random.randint(1, 10))/float(10)
        w = float(weight[j].split('\n')[0])
        p = int(len(rs) * w)
        print(str(p) + ', ', end='')
        E['price'] = [p]
        B += p
        E['rows'] = [len(rs)]
        Es[fnm] = E
        j += 1
    print()
    B = B * 0.5
    print('Budget: ' + str(B))

    um(Es, weights, weights1, options, emb, Fnms, B)




