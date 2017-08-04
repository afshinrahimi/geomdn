'''
Created on 21 Feb 2017
Given a training set of lat/lon as input and probability distribution over words as output,
train a model that can predict words based on location.
then try to visualise borders and regions (e.g. try many lat/lon as input and get the probability of word yinz
in the output and visualise that).
@author: af
'''
import matplotlib as mpl
import re
from itertools import product
mpl.use('Agg')
import matplotlib.mlab as mlab
from matplotlib import ticker
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.mlab import griddata
from matplotlib.patches import Polygon as MplPolygon
#import seaborn as sns
#sns.set(style="white")
import operator
from scipy.stats import multivariate_normal
import argparse
import sys
from scipy.spatial import ConvexHull
import os
import pdb
import random
from data import DataLoader
import numpy as np
import sys
from os import path
import scipy as sp
import theano
import shutil
import theano.tensor as T
import lasagne
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params
import theano.sparse as S
from lasagne.layers import DenseLayer, DropoutLayer
import logging
import json
import codecs
import pickle
import gzip
from collections import OrderedDict, Counter
from sklearn.preprocessing import normalize
from haversine import haversine
from _collections import defaultdict
from scipy import stats
from mpl_toolkits.basemap import Basemap, cm, maskoceans
from scipy.interpolate import griddata as gd
from lasagne_layers import SparseInputDenseLayer, GaussianRBFLayer, BivariateGaussianLayer, MDNSharedParams
from shapely.geometry import MultiPoint, Point, Polygon
import shapefile
from utils import short_state_names, stop_words, get_us_city_name
from sklearn.cluster import KMeans, MiniBatchKMeans
import utils
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)   
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)
np.random.seed(77)
random.seed(77)

def get_us_border_polygon():
    
    sf = shapefile.Reader("./data/states/cb_2015_us_state_20m")
    shapes = sf.shapes()
    #shapes[i].points
    fields = sf.fields
    records = sf.records()
    state_polygons = {}
    for i, record in enumerate(records):
        state = record[5]
        points = shapes[i].points
        poly = Polygon(points)
        state_polygons[state] = poly

    return state_polygons

def geo_latlon_eval(latlon_true, latlon_pred):
    distances = []
    for i in range(0, len(latlon_true)):
        lat_true, lon_true = latlon_true[i]
        lat_pred, lon_pred = latlon_pred[i]
        distance = haversine((lat_true, lon_true), (lat_pred, lon_pred))
        distances.append(distance)
    acc_at_161 = 100 * len([d for d in distances if d < 161]) / float(len(distances))
    logging.info( "Mean: " + str(int(np.mean(distances))) + " Median: " + str(int(np.median(distances))) + " Acc@161: " + str(int(acc_at_161)))
    return np.mean(distances), np.median(distances), acc_at_161

#us border
state_polygons = get_us_border_polygon()   

def in_us(lat, lon):
    p = Point(lon, lat)
    for state, poly in state_polygons.iteritems():
        if poly.contains(p):
            return state
    return None

def inspect_inputs(i, node, fn):
    print(i, node, "input(s) shape(s):", [input[0].shape for input in fn.inputs])
    #print(i, node, "input(s) stride(s):", [input.strides for input in fn.inputs], end='')

def inspect_outputs(i, node, fn):
    print(" output(s) shape(s):", [output[0].shape for output in fn.outputs])
    #print(" output(s) stride(s):", [output.strides for output in fn.outputs])    

def softplus(x):
    return np.log(np.exp(x) + 1)
def softsign(x):
    return x / (1 + np.abs(x))

class NNModel():
    def __init__(self, 
                 n_epochs=10, 
                 batch_size=1000, 
                 regul_coef=1e-6,
                 input_size=None,
                 output_size = None, 
                 hid_size=500, 
                 drop_out=False, 
                 dropout_coef=0.5,
                 early_stopping_max_down=10,
                 dtype='float32',
                 autoencoder=100,
                 reload=False,
                 n_gaus_comp=500,
                 mus=None,
                 sigmas=None,
                 corxy=None,
                 nomdn=False,
                 dataset_name=''):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.regul_coef = regul_coef
        self.hid_size = hid_size
        self.dropout_coef = dropout_coef
        self.early_stopping_max_down = early_stopping_max_down
        self.dtype = dtype
        self.input_size = input_size
        self.output_size = output_size
        self.autoencoder = autoencoder
        self.reload = reload
        self.n_gaus_comp = n_gaus_comp
        self.mus = mus
        self.sigmas = sigmas
        self.corxy = corxy
        self.nomdn = nomdn
        self.dataset_name = dataset_name
        logging.info('building nn model with %d gaussian components and %d hidden layer...' % (self.n_gaus_comp, self.hid_size))
        self.build()
        
        
    def build(self):

        self.X_sym = T.matrix()
        self.Y_sym = T.matrix()

        l_in = lasagne.layers.InputLayer(shape=(None, self.input_size),
                                         input_var=self.X_sym)
        logging.info('adding %d-comp bivariate gaussian layer...' %self.n_gaus_comp)
        self.l_gaus = MDNSharedParams(l_in, num_units=self.n_gaus_comp, mus=self.mus, sigmas=self.sigmas, corxy=self.corxy,
                                      nonlinearity=lasagne.nonlinearities.softmax,
                                      W=lasagne.init.GlorotUniform())

        self.l_hid = lasagne.layers.DenseLayer(self.l_gaus, num_units=self.hid_size, 
                                  nonlinearity=lasagne.nonlinearities.rectify,
                                  W=lasagne.init.GlorotUniform())
        
        pis = lasagne.layers.get_output(self.l_gaus, self.X_sym)
        mus, sigmas, corxy = self.l_gaus.mus, self.l_gaus.sigmas, self.l_gaus.corxy
        sigmas = T.nnet.softplus(sigmas) 
        corxy = T.nnet.nnet.softsign(corxy)
        nll_loss_coef = 1.0
        nll_loss = self.nll_loss_sharedparams(mus, sigmas, corxy, pis, self.X_sym)

                
        self.l_out = lasagne.layers.DenseLayer(self.l_hid, num_units=self.output_size,
                                          nonlinearity=lasagne.nonlinearities.softmax,
                                          W=lasagne.init.GlorotUniform())
        
        
        
        mus_pred = self.pred_sharedparams_sym(mus, sigmas, corxy, pis)
        sq_loss = lasagne.objectives.squared_error(mus_pred, self.X_sym).mean()
        sq_loss_coef = 1.0
        
        regul_loss = lasagne.regularization.regularize_network_params(self.l_out, penalty=lasagne.regularization.l2)
        regul_loss_coef = 1.0

        #enforce sigmas to be lower than k
        k = 3
        #enforce sigma to be smaller than k
        sigma_constrain_loss = T.sum((((sigmas + k) + T.abs_(sigmas - k))/2.0 - k) ** 2)
        sigma_constrain_loss_coef = 1.0
        
        self.gaus_output = pis
        self.eval_output = lasagne.layers.get_output(self.l_out, self.X_sym, deterministic=True)
        self.output = lasagne.layers.get_output(self.l_out, self.X_sym)
        cross_loss = lasagne.objectives.categorical_crossentropy(self.output, self.Y_sym).mean()
        cross_entropy_coef = 1.0
        loss = cross_entropy_coef * cross_loss + nll_loss * nll_loss_coef \
                             + sigma_constrain_loss * sigma_constrain_loss_coef \
                             + sq_loss * sq_loss_coef + regul_loss * regul_loss_coef
        eval_cross_loss = lasagne.objectives.categorical_crossentropy(self.eval_output, self.Y_sym).mean()
        eval_loss = cross_entropy_coef * eval_cross_loss + nll_loss * nll_loss_coef \
                     + sigma_constrain_loss * sigma_constrain_loss_coef\
                     + sq_loss * sq_loss_coef + regul_loss * regul_loss_coef

      
        parameters = lasagne.layers.get_all_params(self.l_out, trainable=True)
        updates = lasagne.updates.adamax(loss, parameters, learning_rate=2e-3, beta1=0.9, beta2=0.999, epsilon=1e-8)
        self.f_gaus = theano.function([self.X_sym], self.gaus_output, on_unused_input='warn')
        self.f_train = theano.function([self.X_sym, self.Y_sym], loss, updates=updates, on_unused_input='warn')
        self.f_val = theano.function([self.X_sym, self.Y_sym], eval_loss, on_unused_input='warn')
        self.f_cross_entropy_loss = theano.function([self.X_sym, self.Y_sym], eval_cross_loss, on_unused_input='warn')
        self.f_predict_proba = theano.function([self.X_sym], self.eval_output, on_unused_input='warn') 
     
    def nll_loss_sharedparams(self, mus, sigmas, corxy, pis, y_true):
        mus_ex = mus[np.newaxis, :, :]
        X = y_true[:, np.newaxis, :]
        diff = X - mus_ex
        diffprod = T.prod(diff, axis=-1)
        corxy2 = corxy **2
        diff2 = diff ** 2
        sigmas2 = sigmas ** 2
        sigmainvs = 1.0 / sigmas
        sigmainvprods = sigmainvs[:, 0] * sigmainvs[:, 1]
        diffsigma = diff2 / sigmas2
        diffsigmanorm = T.sum(diffsigma, axis=-1)
        z = diffsigmanorm - 2 * corxy * diffprod * sigmainvprods
        oneminuscorxy2inv = 1.0 / (1.0 - corxy2)
        expterm = -0.5 * z * oneminuscorxy2inv
        new_exponent = T.log(0.5/np.pi) + T.log(sigmainvprods) + T.log(np.sqrt(oneminuscorxy2inv)) + expterm + T.log(pis)
        max_exponent = T.max(new_exponent ,axis=1, keepdims=True)
        mod_exponent = new_exponent - max_exponent
        gauss_mix = T.sum(T.exp(mod_exponent),axis=1)
        log_gauss = max_exponent + T.log(gauss_mix)
        loss = -T.mean(log_gauss)
        return loss
    def pred_sharedparams_sym(self, mus, sigmas, corxy, pis, prediction_method='mixture'):
        '''
        select mus that maximize \sum_{pi_i * prob_i(mu)} if prediction_method is mixture
        else
        select the component with highest pi if prediction_method is pi.
        '''
        if prediction_method == 'mixture':
            X = mus[:, np.newaxis, :]
            diff = X - mus
            diffprod = T.prod(diff, axis=-1)
            sigmainvs = 1.0 / sigmas
            sigmainvprods = sigmainvs[:, 0] * sigmainvs[:, 1]
            sigmas2 = sigmas ** 2
            corxy2 = corxy **2
            diff2 = diff ** 2
            diffsigma = diff2 / sigmas2
            diffsigmanorm = T.sum(diffsigma, axis=-1)
            z = diffsigmanorm - 2 * corxy * diffprod * sigmainvprods
            oneminuscorxy2inv = 1.0 / (1.0 - corxy2)
            term = -0.5 * z * oneminuscorxy2inv
            expterm = T.exp(term)
            probs = (0.5 / np.pi) * sigmainvprods * T.sqrt(oneminuscorxy2inv) * expterm
            piprobs = pis[:, np.newaxis, :] * probs
            piprobsum = T.sum(piprobs, axis=-1)
            preds = T.argmax(piprobsum, axis=1)
            selected_mus = mus[preds, :]
     
            return selected_mus
        elif prediction_method == 'pi':
            logging.info('only pis are used for prediction')
            preds = T.argmax(pis, axis=1)
            selected_mus = mus[preds, :]      
            return selected_mus
        else:
            raise('%s is not a valid prediction method' %prediction_method)
 

    def set_params(self, params):
        lasagne.layers.set_all_param_values(self.l_out, params)

    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        assert inputs.shape[0] == targets.shape[0]
        if shuffle:
            indices = np.arange(inputs.shape[0])
            np.random.shuffle(indices)
        for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]       
    
    def fit(self, X_train, Y_train, X_dev, Y_dev, X_test, Y_test):
        model_file = './dumps/loc2lang_%s_hid%d_gaus%d_out%d_%s.pkl' %(self.dataset_name, self.hid_size, self.n_gaus_comp, self.output_size, str(self.nomdn))
        if self.reload:
            if path.exists(model_file):
                logging.info('loading the model from %s' %model_file)
                with open(model_file, 'rb') as fin:
                    params = pickle.load(fin)
                self.set_params(params)
                
        else:
            logging.info('training with %d n_epochs and  %d batch_size' %(self.n_epochs, self.batch_size))
            best_params = None
            best_val_loss = sys.maxint
            n_validation_down = 0
            
            #train autoencoder
            
            for step in range(self.n_epochs):
                if step % 10 == 0:    
                    best_params = lasagne.layers.get_all_param_values(self.l_out)
                    visualise_bigaus(params_file=None, params=best_params, iter=step, output_type='png')
                l_trains = []
                for batch in self.iterate_minibatches(X_train, Y_train, self.batch_size, shuffle=True):
                    x_batch, y_batch = batch
                    if sp.sparse.issparse(y_batch): y_batch = y_batch.todense().astype('float32')
                    l_train = self.f_train(x_batch, y_batch)
                    l_trains.append(l_train)
                l_train = np.mean(l_trains)
                l_val = self.f_val(X_dev, Y_dev)
                if l_val < best_val_loss and (best_val_loss - l_val) > (0.0001 * l_val):
                    best_params = lasagne.layers.get_all_param_values(self.l_out)
                    best_val_loss = l_val
                    if not self.nomdn:
                        logging.info('first mu (%f,%f) first covar (%f, %f, %f)' %(best_params[2][0, 0], best_params[2][0, 1], softplus(best_params[3][0, 0]), softplus(best_params[3][0, 1]), softsign(best_params[4][0])))
                        logging.info('second mu (%f,%f) second covar (%f, %f, %f)' %(best_params[2][1, 0], best_params[2][1, 1], softplus(best_params[3][1, 0]), softplus(best_params[3][1, 1]), softsign(best_params[4][1])))
                    n_validation_down = 0
                else:
                    n_validation_down += 1
                    if n_validation_down > self.early_stopping_max_down:
                        logging.info('validation results went down. early stopping ...')
                        break
    
                logging.info('iter %d, train loss %f, dev loss %f, best dev loss %f, num_down %d' %(step, l_train, l_val, best_val_loss, n_validation_down))
            
            lasagne.layers.set_all_param_values(self.l_out, best_params)
            #for debugging the output of gaussian layer
            #debug_gaussian(best_params)
            logging.info('dumping the model in %s' %model_file)
            with open(model_file, 'wb') as fout:
                pickle.dump(best_params, fout)        
        
        l_test = self.f_cross_entropy_loss(X_test, Y_test)
        perplexity_test = np.power(2, l_test)
        logging.info('test loss is %f and perplexity is %f' %(l_test, perplexity_test))
        l_dev = self.f_cross_entropy_loss(X_dev, Y_dev)
        perplexity_dev = np.power(2, l_dev)
        logging.info('dev loss is %f and perplexity is %f' %(l_dev, perplexity_dev))
        

        return perplexity_test, perplexity_dev
                
    def predict(self, X):
        prob_dist = self.f_predict_proba(X)
        return prob_dist 

 

 

          


def get_cluster_centers(input, n_cluster, raw=True):
    '''
    given lat/lons of training samples cluster them
    and find the clusters' mus, sigmas and corxys.
    
    if raw is True then run inverse softplus and softsign on sigmas and corxys
    so that when softplus and softsign is performed on them in the neural network
    the actual sigmas and corxys are recovered.
    '''
    kmns = MiniBatchKMeans(n_clusters=n_cluster, batch_size=1000)
    kmns.fit(input)
    sigmas = np.zeros(shape=(n_cluster, 2), dtype='float32')
    corxys = np.zeros(n_cluster, dtype='float32')
    for i in range(n_cluster):
        indices = np.where(kmns.labels_ == i)[0]
        samples = input[indices]
        #rowvar should be False so that each column is considered a variable (not each row)
        covmat = np.cov(samples, rowvar=False)
        if samples.shape[0] == 1:
            #only one sample in the cluster
            stdlatlat = 1
            stdlonlon = 1
            covlatlon = 0
        else: 
            stdlatlat = np.sqrt(covmat[0, 0])
            stdlonlon = np.sqrt(covmat[1, 1])
            covlatlon = covmat[0, 1]
        if raw:
            #softplus will be applied on sigmas so now apply the reverse so that they become sigmas in neural network
            sigmas[i, 0] = np.log(np.exp(stdlatlat) - 1)
            sigmas[i, 1] = np.log(np.exp(stdlonlon) - 1)
    
    
            corlatlon = covlatlon / (stdlatlat * stdlonlon)
            #do inverse softsign on corlatlon because we later run softsign on corlatlon values in the neural network: softsign = x / (1 + abs(x))
            #later when softsign is applied on unprocessed_cor, corlatlon will be retrieved
            softsigncor = corlatlon/ (1 + np.abs(corlatlon))
            raw_cor = corlatlon / (1.0 - corlatlon * np.sign(softsigncor))
            corxys[i] = raw_cor
        else:
            corxys[i] = corlatlon
            sigmas[i, 0] = stdlatlat
            sigmas[i, 1] = stdlonlon
        
    mus = kmns.cluster_centers_.astype('float32')
    return mus, sigmas, corxys 
    
def get_named_entities(documents, mincount=10):
    '''
    given a list of texts find words that more than 
    50% of time start with a capital letter and return them as NE
    '''
    word_count = defaultdict(int)
    word_capital = defaultdict(int)
    NEs = []
    token_pattern = r'(?u)(?<![#@])\b\w+\b'
    tp = re.compile(token_pattern)
    for doc in documents:
        words = tp.findall(doc)
        for word in words:
            if word[0].isupper():
                word_capital[word.lower()] += 1
            word_count[word.lower()] += 1

    for word, count in word_count.iteritems():
        if count < mincount: continue
        capital = word_capital[word]
        percent = float(capital) / count
        if percent > 0.7:
            NEs.append(word)
    return NEs

def get_dare_words():
    word_dialect = {}
    with open('./data/geodare.cleansed.filtered.json', 'r') as fin:
        for line in fin:
            line = line.strip()
            dialect_word = json.loads(line)
            word_dialect[dialect_word['word']] = dialect_word['dialect'].lower()
    return word_dialect
     
def load_data(data_home, **kwargs):
    bucket_size = kwargs.get('bucket', 300)
    dataset_name = kwargs.get('dataset_name')
    encoding = kwargs.get('encoding', 'utf-8')
    celebrity_threshold = kwargs.get('celebrity', 10)  
    mindf = kwargs.get('mindf', 10)
    dtype = kwargs.get('dtype', 'float32')
    one_hot_label = kwargs.get('onehot', False)
    grid_transform = kwargs.get('grid', False)
    normalize_words = kwargs.get('norm', False)


    vocab = None
    vocab_file = './dumps/na_vocab.pkl'
    if 'na' in dataset_name and path.exists(vocab_file):
        with open(vocab_file, 'rb') as fin:
            vocab = pickle.load(fin)
    extract_vocab = False
    norm = 'l1'
    idf = False
    if extract_vocab:
        norm = None
        idf = False
        
    dl = DataLoader(data_home=data_home, bucket_size=bucket_size, encoding=encoding, 
                    celebrity_threshold=celebrity_threshold, one_hot_labels=one_hot_label, 
                    mindf=mindf, maxdf=0.2, norm=norm, idf=idf, btf=True, tokenizer=None, 
                    subtf=True, stops=stop_words, token_pattern=r'(?u)(?<![@])#?\b\w\w+\b', vocab=vocab)
    logging.info('loading dataset...')
    dl.load_data()

    #load words that often start with uppercase (heuristic named entity detection)
    ne_file = './data/ne_' + dataset_name + '.json'
    if path.exists(ne_file):
        with codecs.open(ne_file, 'r', encoding='utf-8') as fout:
            NEs = json.load(fout)
        NEs = NEs['nes']
    else:
        NEs = get_named_entities(dl.df_train.text.values, mincount=mindf)
        with codecs.open(ne_file, 'w', encoding='utf-8') as fout:
            json.dump({'nes': NEs}, fout)

        
    U_test = dl.df_test.index.tolist()
    U_dev = dl.df_dev.index.tolist()
    U_train = dl.df_train.index.tolist()  
    #locations should be used as input
    loc_test = np.array([[a[0], a[1]] for a in dl.df_test[['lat', 'lon']].values.tolist()], dtype='float32')
    loc_train = np.array([[a[0], a[1]] for a in dl.df_train[['lat', 'lon']].values.tolist()], dtype='float32')
    loc_dev = np.array([[a[0], a[1]] for a in dl.df_dev[['lat', 'lon']].values.tolist()], dtype='float32')
    
    dl.tfidf()
    word_dialect = get_dare_words()


    
    if extract_vocab:
        #words that should be used in the output and be predicted
        w_freq = np.array(dl.X_train.sum(axis=0))[0]
        vocab = dl.vectorizer.get_feature_names()
        vocab_freq = {vocab[i]: w_freq[i] for i in xrange(len(vocab))}
        frequent_dare_words = set()
        frequent_vocab_words = set([vocab[i] for i in xrange(len(vocab)) if w_freq[i] >= 100])
        for w in word_dialect:
            freq = vocab_freq.get(w, 0)
            if freq > 10:
                frequent_dare_words.add(w)
        logging.info('found %d frequent dare words' %len(frequent_dare_words))
        for dare_w in frequent_dare_words:
            frequent_vocab_words.add(dare_w)
        new_vocab = sorted(frequent_vocab_words)
        with open('./dumps/' + dataset_name + '_vocab.pkl', 'wb') as fout:
            pickle.dump(new_vocab, fout)

    
            
    W_train = dl.X_train
    W_dev = dl.X_dev.todense().astype('float32')
    W_test = dl.X_test.todense().astype('float32')
    vocab = dl.vectorizer.get_feature_names()
    data = (loc_train, W_train, loc_dev, W_dev, loc_test, W_test, vocab)
    return data

def state_dialect_words(loc_train, vocab, model, N=1000):
    #sample N training locations
    indices = np.arange(loc_train.shape[0])
    np.random.shuffle(indices)
    random_indices = indices[0:2 * N]
    sampled_locations = loc_train[random_indices, :]
    all_loc_state = utils.get_state_from_coordinates(sampled_locations)
    locs = all_loc_state.keys()
    random.shuffle(locs)
    locs = locs[0: N]
    loc_state = {}
    vocabset = set(vocab)
    state_indices = defaultdict(set)
    dialect_indices = defaultdict(set)
    dialect_states = utils.dialect_state
    state_dialects = defaultdict(set)
    new_dialect_states = defaultdict(set)
    for dialect, states in dialect_states.iteritems():
        dialect = dialect.lower()
        states = set([s.lower() for s in states])
        new_dialect_states[dialect] = states
        for state in states:
            state_dialects[state].add(dialect)
    dialect_states = new_dialect_states
    
    for i, loc in enumerate(locs):
        state = all_loc_state[loc]
        loc_state[loc] = state
        state_indices[state].add(i)
        dialects = state_dialects[state]
        for dialect in dialects:
            dialect_indices[dialect].add(i)
        dialect_indices[state].add(i)
    
    locs = np.array(locs).astype('float32')
    sampled_predictions = model.predict(locs)
    point_dialects = set([state.lower() for state in loc_state.values()])
    #add related dialects for each state
    for state, dls in state_dialects.iteritems():
        for d in dls:
            point_dialects.add(d)
    word_dialect = get_dare_words()
    word_dialect = {w:dialect for w, dialect in word_dialect.iteritems() if w in vocabset}
    dare_dialects = set(word_dialect.values())
    covered_dialects = dare_dialects & point_dialects
    logprobs = np.log(sampled_predictions)
    #logprobs = sampled_predictions
    dialect_count = [(d, len(indices)) for d, indices in dialect_indices.iteritems()]
    logging.info(dialect_count)
    global_mean_logprobs = np.mean(logprobs, axis=0)
    dialect_ranking = {}
    for dialect in covered_dialects:
        dialect_loc_indices = sorted(dialect_indices[dialect])
        dialect_logprobs = logprobs[dialect_loc_indices, :]
        dialect_mean_logprobs = np.mean(dialect_logprobs, axis=0)
        dialect_normalized_logprobs = dialect_mean_logprobs - global_mean_logprobs
        sorted_vocab_indices = np.argsort(dialect_normalized_logprobs)
        sorted_vocab = np.array(vocab)[sorted_vocab_indices].tolist()
        dialect_ranking[dialect] = list(reversed(sorted_vocab))
    printable_dialect_ranking = {d:rank[0:100] for d, rank in dialect_ranking.iteritems()}
    with open('./dumps/dialect_ranking_' + str(len(vocab)) + '.json', 'w') as fout:
        json.dump(printable_dialect_ranking, fout)
    #recall at k for each state
    intervals = [0.01, 0.05, 0.1, 0.15, 0.2]
    #ks = [max(1, int(i * len(vocab))) for i in intervals]
    ks = [int(i * len(vocab)) for i in intervals]
    k_recall = defaultdict(list)
    oracle_k_recall = defaultdict(list)
    for dialect in covered_dialects:
        dialect_dare_words = set([w for w, d in word_dialect.iteritems() if d == dialect])
        retrieved_words = dialect_ranking[dialect]
        oracle_retrieved = list(set(retrieved_words) & dialect_dare_words)
        logging.info('dialect %s DARE worlds in vocab: %d' %(dialect, len(oracle_retrieved)))
        #recall at k
        for k in ks:
            words_at_k = set(retrieved_words[0:k])
            #number of correct retrievals
            correct = len(words_at_k & dialect_dare_words)
            recall_at_k = float(correct) / len(dialect_dare_words)
            k_recall[k].append(recall_at_k)
            
            
            oracle_words_at_k = set(oracle_retrieved[0:k])
            oracle_correct = len(oracle_words_at_k & dialect_dare_words)
            oracle_recall_at_k = float(oracle_correct) / len(dialect_dare_words)
            oracle_k_recall[k].append(oracle_recall_at_k)
    
    for k in ks:
        recalls = k_recall[k]
        oracle_recalls = oracle_k_recall[k]
        logging.info('recall at %d is %f%% oracle %f%%' %(k, np.mean(recalls)*100, np.mean(oracle_recalls)*100))
    
        
        
        
        
        
    
      
    
def train(data, **kwargs):
    dropout_coef = kwargs.get('dropout_coef', 0.5)
    regul = kwargs.get('regul_coef', 1e-6)
    hid_size = kwargs.get('hidden_size', 500)
    autoencoder = kwargs.get('autoencoder', False)
    grid_transform = kwargs.get('grid', False)
    n_gaus_comp = kwargs.get('ncomp', 500)
    dataset_name = kwargs.get('dataset_name')
    kmeans_mu = kwargs.get('kmeans', True)
    nomdn = kwargs.get('nomdn', False)
    tune = kwargs.get('tune', False)
    loc_train, W_train, loc_dev, W_dev, loc_test, W_test, vocab = data
    input_size = loc_train.shape[1]
    output_size = W_train.shape[1]
    batch_size = kwargs.get('batch', 1000)

    
    
    mus = None
    raw_stds = None
    raw_cors = None
    
    if not nomdn:
        if kmeans_mu:
            logging.info('initializing mus, sigmas and corxy by clustering training points')
            mus, raw_stds, raw_cors = get_cluster_centers(loc_train, n_cluster=n_gaus_comp)
            logging.info('first mu is %s' %str(mus[0, :]))
        else:
            logging.info('initializing mus by n random training samples...')
            #set all mus to center of US
            indices = np.arange(loc_train.shape[0])
            np.random.shuffle(indices)
            random_indices = indices[0:n_gaus_comp]
            mus = loc_train[random_indices, :]
            set_to_center = False
            if set_to_center:
                logging.info('set all mus to the center of USA with a little noise')
                for i in range(mus.shape[0]):
                    mus[i, 0] = 39.5 + np.random.uniform(low=-3, high=+3)
                    mus[i, 1] = -98.35 + np.random.uniform(low=-3, high=+3)
        mus = mus.astype('float32')
        raw_stds = None
        raw_cors = None
    
    model = NNModel(n_epochs=10000, batch_size=batch_size, regul_coef=regul, 
                    input_size=input_size, output_size=output_size, hid_size=hid_size, 
                    drop_out=False, dropout_coef=dropout_coef, early_stopping_max_down=3, 
                    autoencoder=autoencoder, reload=True, n_gaus_comp=n_gaus_comp, mus=mus, 
                    sigmas=raw_stds, corxy=raw_cors, nomdn=nomdn, dataset_name=dataset_name)
    #pdb.set_trace()
    perplexity_test, perplexity_dev = model.fit(loc_train, W_train, loc_dev, W_dev, loc_test, W_test)
    #model.fit(loc_train, loc_train, loc_dev, loc_dev, loc_test, loc_test)
    

    if tune:
        return perplexity_test, perplexity_dev
    
    #load named entities
    ne_file = './dumps/ne_' + dataset_name + '.json'
    with codecs.open(ne_file, 'r', encoding='utf-8') as fout:
        NEs = json.load(fout)
    NEs = set(NEs['nes'])
    
    k = 150
    with open('./data/cities.json', 'r') as fin:
        cities = json.load(fin)
    local_word_file = './dumps/local_words_'  + str(W_train.shape)+ '_' + str(n_gaus_comp) + '.txt'
    with codecs.open(local_word_file, 'w', encoding='utf-8') as fout:
        cities = cities[0:100]
        for city in cities:
            name = city['city']
            lat, lon = city['latitude'], city['longitude']
            loc = np.array([[lat, lon]]).astype('float32')
            preds = model.predict(loc)
            topword_indices = np.argsort(preds)[0][::-1][:k]
            topwords = [vocab[i] for i in topword_indices]
            #check if a topword is a named entity add a star beside it
            newtopwords = []
            for topword in topwords:
                if topword in NEs:
                    topword = topword + "_NE"
                newtopwords.append(topword)
            #logging.info(name)
            #logging.info(str(topwords))
            fout.write('\n*****%s*****\n' %name)
            fout.write(str(newtopwords))
            
    
    
    # us bounding box (-124.848974, 24.396308) - (-66.885444, 49.384358)
    lllat = 24.396308
    lllon = -124.848974
    urlat =  49.384358
    urlon = -66.885444
    step = 0.5
    if dataset_name == 'world-final':
        lllat = -90
        lllon = -180
        urlat = 90
        urlon = 180
        step = 0.5
    lats = np.arange(lllat, urlat, step)
    lons = np.arange(lllon, urlon, step)
    
    check_in_us = True if dataset_name != 'world-final' else False
    
    if check_in_us:
        coords = []
        for lat in lats:
            for lon in lons:
                if in_us(lat, lon):
                    coords.append([lat, lon])
                
        logging.info('%d coords within continental US' %len(coords))
        coords = np.array(coords).astype('float32')
    else:
        coords = np.array(map(list, product(lats, lons))).astype('float32')


    preds = model.predict(coords)
    info_file = './dumps/coords-preds-vocab' + str(W_train.shape[0])+ '_' + str(n_gaus_comp) + '.pkl'
    logging.info('dumping the results in %s' %info_file)
    with open(info_file, 'wb') as fout:
        pickle.dump((coords, preds, vocab), fout)

    contour_me(info_file, dataset_name=dataset_name)
    
    state_dialect_words(loc_train, vocab, model, N=10000 if dataset_name=='na' else 2000)       
    
def get_local_words(preds, vocab, NEs=[], k=50):
    #normalize the probabilites of each vocab
    normalized_preds = normalize(preds, norm='l1', axis=0)
    entropies = stats.entropy(normalized_preds)
    sorted_indices = np.argsort(entropies)
    sorted_local_words = np.array(vocab)[sorted_indices].tolist()
    filtered_local_words = []
    NEset = set(NEs)
    for word in sorted_local_words:
        if word in NEset: continue
        filtered_local_words.append(word)
    return filtered_local_words[0:k]
   
def contour_me(info_file='./dumps/coords-preds-vocab5685_50.pkl', **kwargs):
    dataset_name = kwargs.get('dataset_name')
    lllat = 24.396308
    lllon = -124.848974
    urlat =  49.384358
    urlon = -66.885444
    if dataset_name == 'world-final':
        lllat = -90
        lllon = -180
        urlat = 90
        urlon = 180
        
    grid_interpolation_method = 'cubic'
    logging.info('interpolation: ' + grid_interpolation_method)
    region_words = {
    "the north":['braht','breezeway','bubbler','clout','davenport','euchre','fridge','hotdish','paczki','pop','sack','soda','toboggan','Yooper'],
    "northeast":['brook','cellar','sneaker','soda'],
    "New England":['grinder','packie','rotary','wicked'],
    "Eastern New England":['bulkhead','Cabinet','frappe','hosey','intervale','jimmies','johnnycake','quahog','tonic'],
    "Northern New England":['ayuh','creemee','dooryard','logan','muckle'],
    "The Mid-Atlantic":['breezeway','hoagie','jawn','jimmies','parlor','pavement','shoobie','youze'],
    "New York City Area":['bodega','dungarees','potsy','punchball','scallion','stoop','wedge'],
    "The Midland":['hoosier'],
    "The South":['banquette','billfold','chuck','commode','lagniappe','yankee','yonder'],
    "The West":['davenport','Hella','snowmachine' ]
    }
    
    word_dialect = {}
    with open('./data/geodare.cleansed.filtered.json', 'r') as fin:
        for line in fin:
            line = line.strip()
            dialect_word = json.loads(line)
            word_dialect[dialect_word['word']] = dialect_word['dialect']
    
            
    map_dir = './maps/' + info_file.split('/')[-1].split('.')[0] + '/'
    if os.path.exists(map_dir):
        shutil.rmtree(map_dir)
    os.mkdir(map_dir)
    
    
    topk_words = []    
    topk_words.extend(word_dialect.keys())
    dialect_words = ['hella', 'yall', 'jawn', 'paczki', 'euchre', 'brat', 'toboggan', 'brook', 'grinder', 'yinz', 'youze', 'yeen']
    topk_words.extend(dialect_words)
    custom_words = ['springfield', 'columbia', 'nigga', 'niqqa', 'bamma', 'cooter', 'britches', 'yapper', 'younguns', 'hotdish', 
                    'schnookered', 'bubbler', 'betcha', 'dontcha']
    topk_words.extend(custom_words)
            
    logging.info('reading info...')
    with open(info_file, 'rb') as fin:
        coords, preds, vocab = pickle.load(fin)
    vocabset = set(vocab)
    dare_in_vocab = set(word_dialect.keys()) & vocabset
    logging.info('%d DARE words, %d in vocab' %(len(word_dialect), len(dare_in_vocab)))
    add_local_words = True
    if add_local_words:
        ne_file = './dumps/ne_' + dataset_name + '.json'
        with codecs.open(ne_file, 'r', encoding='utf-8') as fout:
            NEs = json.load(fout)
        NEs = NEs['nes']
        local_words = get_local_words(preds, vocab, NEs=NEs, k=500)
        logging.info(local_words)
        topk_words.extend(local_words[0:20])
    
    add_cities = False
    if add_cities:
        with open('./data/cities.json', 'r') as fin:
            cities = json.load(fin)
        cities = cities[0:100]
        for city in cities:
            name = city['city'].lower()
            topk_words.append(name)
    wi = 0
    for word in topk_words:
        if word in vocabset:
            fig = plt.figure(figsize=(5, 4))
            grid_transform = kwargs.get('grid', False)
            ax = fig.add_subplot(111, axisbg='w', frame_on=False)
            logging.info('%d mapping %s' %(wi, word))
            wi += 1
            index = vocab.index(word)
            scores = np.log(preds[:, index])
            
            m = Basemap(llcrnrlat=lllat,
            urcrnrlat=urlat,
            llcrnrlon=lllon,
            urcrnrlon=urlon,
            resolution='i', projection='cyl')
            '''
            m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95, resolution='i')
            '''
            m.drawmapboundary(fill_color = 'white')
            #m.drawcoastlines(linewidth=0.2)
            m.drawcountries(linewidth=0.2)
            if dataset_name != 'world-fianl':
                m.drawstates(linewidth=0.2, color='lightgray')
            #m.fillcontinents(color='white', lake_color='#0000ff', zorder=2)
            #m.drawrivers(color='#0000ff')
            #m.drawlsmask(land_color='gray',ocean_color="#b0c4de", lakes=True)
            #m.drawcounties()
            shp_info = m.readshapefile('./data/us_states_st99/st99_d00','states',drawbounds=True, zorder=0)
            printed_names = []
            ax = plt.gca()
            ax.xaxis.set_visible(False) 
            ax.yaxis.set_visible(False) 
            for spine in ax.spines.itervalues(): 
                spine.set_visible(False) 

            state_names_set = set(short_state_names.values())
            mi_index = 0
            wi_index = 0
            for shapedict,state in zip(m.states_info, m.states):
                if dataset_name == 'world-final': break
                draw_state_name = True
                if shapedict['NAME'] not in state_names_set: continue
                short_name = short_state_names.keys()[short_state_names.values().index(shapedict['NAME'])]
                if short_name in printed_names and short_name not in ['MI', 'WI']: 
                    continue
                if short_name == 'MI':
                    if mi_index != 3:
                        draw_state_name = False
                    mi_index += 1
                if short_name == 'WI':
                    if wi_index != 2:
                        draw_state_name = False
                    wi_index += 1
                    
                # center of polygon
                x, y = np.array(state).mean(axis=0)
                hull = ConvexHull(state)
                hull_points = np.array(state)[hull.vertices]
                x, y = hull_points.mean(axis=0)
                if short_name == 'MD':
                    y = y - 0.5
                    x = x + 0.5
                elif short_name == 'DC':
                    y = y + 0.1
                elif short_name == 'MI':
                    x = x - 1
                elif short_name == 'RI':
                    x = x + 1
                    y = y - 1
                #poly = MplPolygon(state,facecolor='lightgray',edgecolor='black')
                #x, y = np.median(np.array(state), axis=0)
                # You have to align x,y manually to avoid overlapping for little states
                if draw_state_name:
                    plt.text(x+.1, y, short_name, ha="center", fontsize=8)
                #ax.add_patch(poly)
                #pdb.set_trace()
                printed_names += [short_name,] 
            mlon, mlat = m(*(coords[:,1], coords[:,0]))
            # grid data
            numcols, numrows = 1000, 1000
            xi = np.linspace(mlon.min(), mlon.max(), numcols)
            yi = np.linspace(mlat.min(), mlat.max(), numrows)

            xi, yi = np.meshgrid(xi, yi)
            # interpolate
            x, y, z = mlon, mlat, scores
            #pdb.set_trace()
            #zi = griddata(x, y, z, xi, yi)
            zi = gd(
                (mlon, mlat),
                scores,
                (xi, yi),
                method=grid_interpolation_method, rescale=False)

            #Remove the lakes and oceans
            data = maskoceans(xi, yi, zi)
            con = m.contourf(xi, yi, data, cmap=plt.get_cmap('YlOrRd'))
            #con = m.contour(xi, yi, data, 3, cmap=plt.get_cmap('YlOrRd'), linewidths=1)
            #con = m.contour(x, y, z, 3, cmap=plt.get_cmap('YlOrRd'), tri=True, linewidths=1)
            #conf = m.contourf(x, y, z, 3, cmap=plt.get_cmap('coolwarm'), tri=True)
            cbar = m.colorbar(con,location='right',pad="2%")
            #plt.setp(cbar.ax.get_yticklabels(), visible=False)
            #cbar.ax.tick_params(axis=u'both', which=u'both',length=0)
            #cbar.ax.set_yticklabels(['low', 'high'])
            tick_locator = ticker.MaxNLocator(nbins=9)
            cbar.locator = tick_locator
            cbar.update_ticks()
            cbar.ax.tick_params(labelsize=11) 
            cbar.ax.yaxis.set_tick_params(pad=2)
            cbar.set_label('logprob', size=11)
            for line in cbar.lines: 
                line.set_linewidth(10)
            
            #read countries for world dataset with more than 100 number of users
            with open('./data/country_count.json', 'r') as fin:
                top_countries = set(json.load(fin))
            world_shp_info = m.readshapefile('./data/CNTR_2014_10M_SH/Data/CNTR_RG_10M_2014','world',drawbounds=False, zorder=100)
            for shapedict,state in zip(m.world_info, m.world):
                if dataset_name != 'world-final':
                    if shapedict['CNTR_ID'] not in ['CA', 'MX']: continue
                else:
                    if shapedict['CNTR_ID'] in top_countries: continue
                poly = MplPolygon(state,facecolor='gray',edgecolor='gray')
                ax.add_patch(poly)
            #plt.title('term: ' + word )
            plt.tight_layout()
            plt.savefig(map_dir + word + '_' + grid_interpolation_method +  '.pdf', bbox_inches='tight')
            plt.close()
            del m


def visualise_bigaus(params_file, params=None, iter=None, output_type='pdf', **kwargs):
    if params == None:
        with open(params_file, 'rb') as fin:
            params = pickle.load(fin)
            
    mus, sigmas, corxys = params[2], params[3], params[4] 
    dataset_name = kwargs.get('dataset_name')
    lllat = 24.396308
    lllon = -124.848974
    urlat =  49.384358
    urlon = -66.885444

    if dataset_name == 'world-final':
        lllat = -90
        lllon = -180
        urlat = 90
        urlon = 180
    fig = plt.figure(figsize=(4, 2.5))
    ax = fig.add_subplot(111, axisbg='w', frame_on=False)
    m = Basemap(llcrnrlat=lllat,
    urcrnrlat=urlat,
    llcrnrlon=lllon,
    urcrnrlon=urlon,
    resolution='c', projection='cyl')
    
    m.drawmapboundary(fill_color = 'white')
    #m.drawcoastlines(linewidth=0.2)
    m.drawcountries(linewidth=0.2)
    m.drawstates(linewidth=0.2, color='lightgray')
    #m.fillcontinents(color='white', lake_color='#0000ff', zorder=2)
    #m.drawrivers(color='#0000ff')
    m.drawlsmask(land_color='gray',ocean_color="#b0c4de", lakes=True)
    lllon, lllat = m(lllon, lllat)
    urlon, urlat = m(urlon, urlat)
    mlon, mlat = m(*(mus[:,1], mus[:,0]))
    numcols, numrows = 1000, 1000
    X = np.linspace(mlon.min()-2, urlon, numcols)
    Y = np.linspace(lllat, urlat, numrows)

    X, Y = np.meshgrid(X, Y)
    m.scatter(mlon, mlat, s=0.2, c='red')
    
    shp_info = m.readshapefile('./data/us_states_st99/st99_d00','states',drawbounds=True, zorder=0)
    printed_names = []
    ax = plt.gca()
    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False) 
    for spine in ax.spines.itervalues(): 
        spine.set_visible(False) 
    
    state_names_set = set(short_state_names.values())
    mi_index = 0
    wi_index = 0
    for shapedict,state in zip(m.states_info, m.states):
        if dataset_name == 'world-final': break
        draw_state_name = True
        if shapedict['NAME'] not in state_names_set: continue
        short_name = short_state_names.keys()[short_state_names.values().index(shapedict['NAME'])]
        if short_name in printed_names and short_name not in ['MI', 'WI']: 
            continue
        if short_name == 'MI':
            if mi_index != 3:
                draw_state_name = False
            mi_index += 1
        if short_name == 'WI':
            if wi_index != 2:
                draw_state_name = False
            wi_index += 1
            
        # center of polygon
        x, y = np.array(state).mean(axis=0)
        hull = ConvexHull(state)
        hull_points = np.array(state)[hull.vertices]
        x, y = hull_points.mean(axis=0)
        if short_name == 'MD':
            y = y - 0.5
            x = x + 0.5
        elif short_name == 'DC':
            y = y + 0.1
        elif short_name == 'MI':
            x = x - 1
        elif short_name == 'RI':
            x = x + 1
            y = y - 1
        #poly = MplPolygon(state,facecolor='lightgray',edgecolor='black')
        #x, y = np.median(np.array(state), axis=0)
        # You have to align x,y manually to avoid overlapping for little states
        if draw_state_name:
            plt.text(x+.1, y, short_name, ha="center", fontsize=5)
        #ax.add_patch(poly)
        #pdb.set_trace()
        printed_names += [short_name,] 
    
    for k in xrange(mus.shape[0]):
        #here x is longitude and y is latitude
        #apply softplus to sigmas (to make them positive)
        sigmax=np.log(1 + np.exp(sigmas[k][1]))
        sigmay=np.log(1 + np.exp(sigmas[k][0]))
        mux=mlon[k]
        muy=mlat[k]
        corxy = corxys[k]
        #apply the soft sign
        corxy = corxy / (1 + np.abs(corxy))
        #now given corxy find sigmaxy
        sigmaxy = corxy * sigmax * sigmay
        
        #corxy = 1.0 / (1 + np.abs(sigmaxy))
        Z = mlab.bivariate_normal(X, Y, sigmax=sigmax, sigmay=sigmay, mux=mux, muy=muy, sigmaxy=sigmaxy)

        #Z = maskoceans(X, Y, Z)
        

        #con = m.contour(X, Y, Z, levels=[0.0015], linewidths=0.5, colors='darkorange', antialiased=True)
        con = m.contour(X, Y, Z, 0, linewidths=0.5, colors='darkorange', antialiased=True)
        '''
        num_levels = len(con.collections)
        if num_levels > 1:
            for i in range(0, num_levels):
                if i != (num_levels-1):
                    con.collections[i].set_visible(False)
        '''
        contour_labels = False
        if contour_labels:
            plt.clabel(con, [con.levels[-1]], inline=True, fontsize=10)
        
    '''
    world_shp_info = m.readshapefile('./data/CNTR_2014_10M_SH/Data/CNTR_RG_10M_2014','world',drawbounds=False, zorder=100)
    for shapedict,state in zip(m.world_info, m.world):
        if shapedict['CNTR_ID'] not in ['CA', 'MX']: continue
        poly = MplPolygon(state,facecolor='gray',edgecolor='gray')
        ax.add_patch(poly)
    '''                
    if iter:
        iter = str(iter).zfill(3)
    else:
        iter = ''
    plt.tight_layout()
    plt.savefig('./maps/video/gaus_' + iter  + '.' + output_type, frameon=False, dpi=200)
    plt.close()

def tune(data, dataset_name, args, num_iter=100):
    logging.info('tuning over %s' %dataset_name)
    param_scores = []
    random.seed()
    for i in xrange(num_iter):
        logging.info('tuning iter %d' %i)
        np.random.seed(77)
        hidden_size = random.choice([300, 600, 900])
        ncomp = random.choice([250, 500, 1000])
        if args.nomdn:
            ncomp = 0
        logging.info('hidden %d ncomp %d' %(hidden_size, ncomp))
        try:
            perplexity_test, perplexity_dev = train(data, regul_coef=args.regularization, dropout_coef=args.dropout, 
                  hidden_size=hidden_size, autoencoder=args.autoencoder, ncomp=ncomp, dataset_name=dataset_name, tune=True, nomdn=args.nomdn)

        except:
            logging.info('exception occurred')
            continue

        scores = OrderedDict()
        scores['perplexity_test'], scores['perplexity_dev'] = perplexity_test, perplexity_dev
        params = OrderedDict()
        params['hidden'], params['ncomp'] =  hidden_size, ncomp
        param_scores.append([params, scores])
        logging.info(params)
        logging.info(scores)
    for param_score in param_scores:
        logging.info(param_score)
   
def parse_args(argv):
    """
    Parse commandline arguments.
    Arguments:
        argv -- An argument list without the program name.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument( '-i','--dataset', metavar='str',  help='dataset for dialectology',  type=str, default='na')
    parser.add_argument( '-bucket','--bucket', metavar='int',  help='discretisation bucket size',  type=int, default=300)
    parser.add_argument( '-batch','--batch', metavar='int',  help='SGD batch size',  type=int, default=1000)
    parser.add_argument( '-hid','--hidden', metavar='int',  help='Hidden layer size after bigaus layer',  type=int, default=500)
    parser.add_argument( '-mindf','--mindf', metavar='int',  help='minimum document frequency in BoW',  type=int, default=10)
    parser.add_argument( '-d','--dir', metavar='str',  help='home directory',  type=str, default='./data')
    parser.add_argument( '-enc','--encoding', metavar='str',  help='Data Encoding (e.g. latin1, utf-8)',  type=str, default='utf-8')
    parser.add_argument( '-reg','--regularization', metavar='float',  help='regularization coefficient)',  type=float, default=1e-6)
    parser.add_argument( '-drop','--dropout', metavar='float',  help='dropout coef default 0.5',  type=float, default=0.5)
    parser.add_argument( '-cel','--celebrity', metavar='int',  help='celebrity threshold',  type=int, default=10)
    parser.add_argument( '-conv', '--convolution', action='store_true',  help='if true do convolution')
    parser.add_argument( '-map', '--map', action='store_true',  help='if true just draw maps from pre-trained model')
    parser.add_argument( '-tune', '--tune', action='store_true',  help='if true tune the hyper-parameters') 
    parser.add_argument( '-tf', '--tensorflow', action='store_true',  help='if exists run with tensorflow') 
    parser.add_argument( '-autoencoder', '--autoencoder', type=int,  help='the number of autoencoder steps before training', default=0) 
    parser.add_argument( '-grid', '--grid', action='store_true',  help='if exists transforms the input from lat/lon to distance from grids on map')  
    parser.add_argument( '-ncomp', type=int,  help='the number of bivariate gaussians after the input layer', default=500) 
    parser.add_argument( '-m', '--message', type=str) 
    parser.add_argument( '-vbi', '--vbi', type=str,  help='if exists load params from vbi file and visualize bivariate gaussians on a map', default=None)
    parser.add_argument( '-nomdn', '--nomdn', action='store_true',  help='if true use tanh layer instead of MDN') 
    args = parser.parse_args(argv)
    return args
if __name__ == '__main__':
    #THEANO_FLAGS='device=cpu' nice -n 10 python loc2lang.py -d ~/datasets/na/processed_data/ -enc utf-8 -reg 0 -drop 0.0 -mindf 200 -hid 1000 -ncomp 100 -autoencoder 100 -map
    args = parse_args(sys.argv[1:])
    datadir = args.dir
    dataset_name = datadir.split('/')[-3]
    logging.info('dataset: %s' % dataset_name)
    if args.vbi:
        visualise_bigaus(args.vbi, dataset_name=dataset_name)
    elif args.map:
        contour_me(grid=args.grid, dataset_name=dataset_name)
    else:
        data = load_data(data_home=args.dir, encoding=args.encoding, mindf=args.mindf, grid=args.grid, dataset_name=dataset_name)
        if args.tune:
            tune(data, dataset_name, args, num_iter=100)
        else:
            train(data, regul_coef=args.regularization, dropout_coef=args.dropout, 
                  hidden_size=args.hidden, autoencoder=args.autoencoder, ncomp=args.ncomp, dataset_name=dataset_name, nomdn=args.nomdn, batch=args.batch)
