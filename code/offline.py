import warnings
warnings.filterwarnings('ignore')

from parameters import *
from build_iris import *
from build_sparse_hist import *
from preproc_card import *

import copy

# Ensure only CPU is used in Keras
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

def build(dic, totald, options, budget, iris_model):
    nd = options.maxd

    #load model
    vf = iris_model.get_layer('lambda_7').output
    model_encoding = Model(iris_model.layers[0].input, vf)

    for fid in range(len(dic['Fnms'])):
        # step 1. read data
        cols = list(range(totald))
        fnm = dic['Fnms'][fid][0]
        print(fnm)
        KEYs, CNTs, TYPEs = [dic['Keys'][fid][d] for d in cols], [dic['Cnts'][fid][d] for d in cols], [
            dic['Types'][fid][d] for d in cols]
        sample = reads([dic['Samples'][fid]], TYPEs, cols, options.sample_size)

        print('--------------Part I-----------------')
        # step 2. run CORDs
        if options.run_cords:
            print("Running CORDs..")
            import cords
            colsstr = ','.join([str(c) for c in cols])
            cordpath = 'tmp_0.5/castinfo'
            if not os.path.exists(cordpath):
                os.mkdir(cordpath)
            cords_fnm = cordpath + '/cords' + str(fid) + '.log'
            cords.CORDS(dic['Samples'][fid], cords_fnm, colsstr)
        print(dic['Samples'][fid])

        print("Building summaries using pre-trained " + options.model_fnm + "..")
        chs = []
        with open(cords_fnm, 'r') as fcords:
            cnt1 = int(fcords.readline())
            for i in range(cnt1):
                ln = fcords.readline().split('\t')
                chs += [[[int(ln[0])], int(ln[1]), 1, ln[2].rstrip('\n'), [], []]]
            cnt2 = int(fcords.readline())
            for i in range(cnt2):
                ln = fcords.readline().split('\t')
                chs += [
                    [[int(ln[0]), int(ln[1])], int(ln[2]), float(ln[3]), ln[4].rstrip('\n'), [int(ln[5]), int(ln[6])],
                     [], []]]
            cnt3 = int(fcords.readline())
            for i in range(cnt3):
                ln = fcords.readline().split('\t')
                chs += [[[int(ln[0]), int(ln[1]), int(ln[2])], int(ln[3]), float(ln[4]), ln[5].rstrip('\n'), [], []]]
        chsp = []
        for cid, ch in enumerate(chs):
            if len(ch[0]) < 2:
                continue
            chsp += [ch]
        #chsp = copy.deepcopy(chs)
        feats = [[] for _ in range(len(chsp))]
        total_size = 0
        cnt_base = 0
        cnt_sparse, cnt_iris, cnt_hist = 0, 0, 0

        V, KEYo = {}, {}
        neb = options.neb
        nss = np.array([neb] * len(KEYs)).astype(int)

        print("Computing quantizations..")
        KEYsd, CNTsd = bkt_shrink({}, "", KEYs, CNTs, TYPEs, nss, options.nusecpp, [])
        KEYo[neb] = [parse_keys(KEYsd[id], TYPEs[id]) for id in range(len(KEYsd))]

        print("Reading data..")
        Vo = readr({}, {}, [fnm], TYPEs, nss, cols)

        V[neb] = parse_raw(Vo, KEYo[neb], nss)
        neb = int(neb / 2)
        while neb > 1:
            nss = np.array([neb] * len(KEYs)).astype(int)
            KEYsd, CNTsd = bkt_shrink({}, "", KEYsd, CNTsd, TYPEs, nss, options.nusecpp, [])
            KEYo[neb] = [parse_keys(KEYsd[id], TYPEs[id]) for id in range(len(KEYsd))]
            V[neb] = parse_raw(Vo, KEYo[neb], nss)
            neb = int(neb / 2)
        for cid, ch in enumerate(chsp):
            chcol = np.array([cols.index(c) for c in ch[0]])
            sl = options.nt
            ns = rebucket(np.array(dic['Cnts'][fid]), chcol, sl, options.nb, options.neb, 'Range', 'freq')
            ns = np.minimum(ns, options.neb)
            ns, chcol = map(list, zip(*sorted(zip(ns, chcol), reverse=True)))
            ch[-2], ch[-1] = ns, chcol
            Vr = [[] for _ in range(len(ns))]
            for i in range(len(ns)):
                Vr[i] += [l[chcol[i]] for l in V[int(ns[i])]]
            X = prep_test(np.array(Vr).transpose(), np.arange(len(ns)), ns)
            ch[3] = 'Iris'
            xx = gen_test(options.ncd, nd, X, ns, options.normlen, options.nt, options.nm, options.neb,
                                  options.maxd)
            xx = np.concatenate((xx, xx), axis=0)
            feat = model_encoding.predict(xx.reshape(1, 2 * (options.ncd + 2), options.maxd + 1))[0]
            feats[cid] = feat
        #for cid, ch in enumerate(chsp):
            #if len(ch[0])>1:
                #print('\tColumnset ' + (str(ch[0][0])+','+str(ch[0][1])).ljust(25) + '\tw/ #DV ' + str(ch[1]) + ',\tcorr. score ' + str(ch[2]) + '\tbuilt using ' + ch[3])
    #print('Storage budget: ' + str(options.storage) + 'KB, total used size: ' + str(total_size) + 'KB')
    #print('Base ' + str(cnt_base) + ', Sparse ' + str(cnt_sparse) + ', Hist '  + str(cnt_hist) + ', Iris ' + str(cnt_iris))

        print('--------------Part II-----------------')
        dic_feat = {}
        dic_feat['sample'] = sample
        dic_feat['feat'] = feats
        dic_feat['TYPEs'] = TYPEs
        dic_feat['ch'] = chsp
        pickle.dump(dic_feat, open('tmp/feature-' + os.path.splitext(os.path.basename(fnm))[0] + '-' + str(options.storage) + '.pkl', 'wb'))
        dic_bucket = {}
        dic_bucket['KEYo'] = KEYo
        pickle.dump(dic_bucket, open('tmp/bucket-' + os.path.splitext(os.path.basename(fnm))[0] + '-' + str(options.storage) + '.pkl' if len(options.nusebucket)==0 else options.nusebucket, 'wb'))

if __name__ == '__main__':
    options = parse_arg()
    if not os.path.exists('tmp'):
        os.mkdir('tmp')

    dic = readd(options.data_dir + '/' + options.input_fnm, sample_rate=1, nqs=0)

    # Turn storage from how many X of that is used by a production system to actual KBs
    # overhead: options.neb(xi) B/col bucket boundaries, options.sample_size*4/1024 KB/col small sample
    # and 0.5KB base histogram (options.neb bins, counted already during construction)
    options.max_atom_budget *= options.storage
    if options.storage < 0.5:
        options.sample_size *= 0.5
    print('Storage budget ' + str(options.storage * 4 * len(dic['Types'][0])) + 'KB, max atom budget ' + str(4*options.max_atom_budget/1024) + 'KB, sample size ' + str(options.sample_size) + ' rows')
    options.storage = len(dic['Types'][0]) * (options.storage * 4 - options.neb/1024 - options.sample_size/256)
    print('Storage budget exlucding overhead ' + str(options.storage) + 'KB') 

    print("Extracting embedding weights from pre-trained model (can be cached offline)..")
    weights = {}
    iris_full_model = load_model(options.model_fnm).layers[-2]
    weights['Embedding'] = iris_full_model.layers[3].get_weights()
    for i in range(len(iris_full_model.layers)):
        if isinstance(iris_full_model.layers[i], keras.layers.Dense):
            weights[iris_full_model.layers[i].name] = iris_full_model.layers[i].get_weights()

    if options.extract_emb:
        extract_emb(iris_full_model, options.model_fnm, options.neb, options.ncd)

    build(dic, len(dic['DimR'][0]), options, options.storage, iris_full_model)



