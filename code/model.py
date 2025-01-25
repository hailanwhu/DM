import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Dense, Embedding, Lambda, Concatenate, multiply, Conv1D, Reshape, BatchNormalization, Dropout, LeakyReLU, RepeatVector, multiply, Activation, MaxPooling1D, Add
from keras.models import Model, load_model
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint


class WarmUpLearningRateScheduler(Callback):
    def __init__(self, warmup_batches, init_lr, verbose=0):
        super(WarmUpLearningRateScheduler, self).__init__()
        self.warmup_batches = warmup_batches
        self.init_lr = init_lr
        self.verbose = verbose
        self.batch_count = 0
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.batch_count = self.batch_count + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        if self.batch_count <= self.warmup_batches:
            lr = self.batch_count*self.init_lr/self.warmup_batches
            K.set_value(self.model.optimizer.lr, lr)
            if self.verbose > 0:
                print('\nBatch %05d: WarmUpLearningRateScheduler setting learning '
                      'rate to %s.' % (self.batch_count + 1, lr))

def get_sum_model(nr, ncd, nd, neb, ne, nx, nfc = 128, nn=3):
    # nr: embedding size; ncd: compressed \ell; nd: maximum sketch dimensions;
    # neb: max column resolution (128); ne: individual cell embedding length (64);
    # nfc: FC layers size; nn: FC layers

    ndr = 0.3
    i = Input(shape=(ncd*2 + 4, nd + 1,))
    xv = Lambda(lambda x: K.repeat_elements(K.expand_dims(x[:, :, 0]), nr, axis=2))(i) # increase one dimension, size 128
    x = Lambda(lambda x: x[:, :, 1:])(i)
    print(x.shape, ncd, nd)

    x = Reshape(((ncd*2 + 4) * nd, 1))(x)
    x = Embedding(neb + 1 + 1 + 1 + 1, ne, mask_zero=False, name='phic')(x)
    x = Lambda(lambda x: x, output_shape=lambda s: s)(x)
    x = Reshape(((ncd*2 + 4), nd * ne))(x)
    for n in range(nn):
        x = Dense(nfc, use_bias=True, name='phir' + str(n))(x)
        x = LeakyReLU()(x)
        x = Dropout(ndr)(x)
    x = Dense(nr, name='phirf')(x)
    #x = Activation('elu')(x)
    x = LeakyReLU()(x)
    x = Dropout(ndr)(x)

    st = 0
    ed = ncd

    qr = Lambda(lambda x: x[:, -4:-2, :])(x)
    qr = Reshape((1, 2 * nr))(qr)
    vq = Concatenate(axis=-1)([qr, qr])
    for a in range(nx - 2):
        vq = Concatenate(axis=-1)([vq, qr])

    xk = Lambda(lambda x: x[:, st:ed, :])(x)
    xf = Lambda(lambda x: x[:, st:ed, :])(xv)
    xk = multiply([xk, xf])
    e = Lambda(lambda x: K.sum(x[:, :], axis=1), output_shape=(lambda shape: (shape[0], shape[2])))(xk)
    vd = Concatenate(axis=-1)([e, e])
    for a in range(nx-2):
        vd = Concatenate(axis=-1)([vd, e])

    st = ncd
    ed = 2*ncd
    xk = Lambda(lambda x: x[:, st:ed, :])(x)
    xf = Lambda(lambda x: x[:, st:ed, :])(xv)
    xk = multiply([xk, xf])

    qr = Lambda(lambda x: x[:, -2:, :])(x)
    qr = Reshape((1, 2 * nr))(qr)
    for a in range(nx):
        vq = Concatenate(axis=-1)([vq, qr])

    e = Lambda(lambda x: K.sum(x[:, :], axis=1), output_shape=(lambda shape: (shape[0], shape[2])))(xk)
    for a in range(nx):
        vd = Concatenate(axis=-1)([vd, e])


    for n in range(nn):
        vd = Dense(nr*nx, use_bias=True, name='phis' + str(n))(vd)
        vd = LeakyReLU()(vd)
        vd = Dropout(ndr)(vd)
    vd = Dense(nr*nx, name='phisf')(vd)
    vd = LeakyReLU()(vd)
    vd = Dropout(ndr)(vd)

    for n in range(nn):
        vq = Dense(2*nr*nx, use_bias=True, name='phip' + str(n))(vq)
        vq = LeakyReLU()(vq)
        vq = Dropout(ndr)(vq)
    vq = Dense(2*nr*nx, name='phipf')(vq)
    vq = LeakyReLU()(vq)
    vq = Dropout(ndr)(vq)

    for n in range(nn):
        vd = Dense(nfc, use_bias=True, name='dense1' + str(n))(vd)
        vd = LeakyReLU()(vd)
        vd = Dropout(ndr)(vd)
    vd = Dense(nfc, use_bias=True, name='dense1f')(vd)
    vd = LeakyReLU()(vd)
    vd = Dropout(ndr)(vd)

    for n in range(nn):
        vq = Dense(nfc, use_bias=True, name='dense2' + str(n))(vq)
        vq = LeakyReLU()(vq)
        vq = Dropout(ndr)(vq)
    vq = Dense(nfc, use_bias=True, name='dense2f')(vq)
    vq = LeakyReLU()(vq)
    vq = Dropout(ndr)(vq)
    vq = Lambda(lambda x: x[:, 0, :], output_shape=(lambda shape: (shape[0], shape[2])))(vq)

    cr = multiply([vd, vq])
    cr = Dense(nfc, use_bias=True, name='dense31')(cr)
    cr = LeakyReLU()(cr)
    cr = Dropout(ndr)(cr)
    cr = Dense(nfc, use_bias=True, name='dense32')(cr)
    cr = LeakyReLU()(cr)
    cr = Dropout(ndr)(cr)

    cr = Dense(1, name='dense33')(cr)

    counter = Model(i, cr)
    return counter

def get_query_model(nr, nx, nfc = 128, nn=3):
    ndr = 0.3
    sketch = Input(shape=(1, nr*nx))
    predicate_emb = Input(shape=(1, 2 * nfc*nx))

    vd = sketch
    vq = predicate_emb

    for n in range(nn):
        vd = Dense(nfc, use_bias=True, name='dense1' + str(n))(vd)
        vd = LeakyReLU()(vd)
        vd = Dropout(ndr)(vd)
    vd = Dense(nfc, use_bias=True, name='dense1f')(vd)
    vd = LeakyReLU()(vd)
    vd = Dropout(ndr)(vd)

    for n in range(nn):
        vq = Dense(nfc, use_bias=True, name='dense2' + str(n))(vq)
        vq = LeakyReLU()(vq)
        vq = Dropout(ndr)(vq)
    vq = Dense(nfc, use_bias=True, name='dense2f')(vq)
    vq = LeakyReLU()(vq)
    vq = Dropout(ndr)(vq)
    vq = Lambda(lambda x: x[:, 0, :], output_shape=(lambda shape: (shape[0], shape[2])))(vq)

    cr = multiply([vd, vq])

    cr = Dense(nfc, use_bias=True, name='dense31')(cr)
    cr = LeakyReLU()(cr)
    cr = Dropout(ndr)(cr)
    cr = Dense(nfc, use_bias=True, name='dense32')(cr)
    cr = LeakyReLU()(cr)
    cr = Dropout(ndr)(cr)

    cr = Dense(1, name='dense33')(cr)

    counter = Model([sketch, predicate_emb], cr)
    return counter

def get_data_emb_model(nr, nx, nfc = 128, nn=3):
    ndr = 0.3
    sketch = Input(shape=(1, 2*nr*nx))
    predicate_emb = Input(shape=(1, 4 * nfc * nx))

    vd = sketch
    for n in range(nn):
        vd = Dense(nr*nx, use_bias=True, name='phis' + str(n))(vd)
        vd = LeakyReLU()(vd)
        vd = Dropout(ndr)(vd)
    vd = Dense(nr*nx, name='phisf')(vd)
    vd = LeakyReLU()(vd)
    vd = Dropout(ndr)(vd)

    vq = predicate_emb
    for n in range(nn):
        vq = Dense(2*nr*nx, use_bias=True, name='phip' + str(n))(vq)
        vq = LeakyReLU()(vq)
        vq = Dropout(ndr)(vq)
    vq = Dense(2*nr*nx, name='phipf')(vq)
    vq = LeakyReLU()(vq)
    vq = Dropout(ndr)(vq)

    emb = Model([sketch, predicate_emb], [vd, vq])
    return emb


def get_merge_emb_model(nr, nfc = 128, nn=3):
    # nr: embedding size;
    # nfc: FC layers size; nn: FC layers

    ndr = 0.3
    i = Input(shape=(1, 2*nr))
    x = Lambda(lambda x: x)(i)

    for n in range(nn):
        x = Dense(nfc, use_bias=True, name='Phi' + str(n))(x)
        x = LeakyReLU()(x)
        x = Dropout(ndr)(x)
    x = Dense(nfc, name='Phif')(x)
    x = LeakyReLU()(x)
    x = Dropout(ndr)(x)

    emb = Model(i, x)

    return emb


