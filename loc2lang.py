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
from matplotlib.patches import Polygon as MplPolygon
import argparse
import sys
from scipy.spatial import ConvexHull
import os
import pdb
import random
from data import DataLoader
import numpy as np
from os import path
import scipy as sp
import theano
import theano.tensor as T
import lasagne
import logging
import json
import codecs
import pickle
from collections import OrderedDict
from sklearn.preprocessing import normalize
from haversine import haversine
from _collections import defaultdict
from scipy import stats
from mpl_toolkits.basemap import Basemap, maskoceans
from scipy.interpolate import griddata as gd
from lasagne_layers import BivariateGaussianLayer
from shapely.geometry import Point, Polygon
import shapefile
from utils import short_state_names, stop_words
from sklearn.cluster import MiniBatchKMeans
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

class Loc2Lang():
    """
    This class implements a NN with 2d location as input
    and a probability distribution over unigram vocabulary
    as output.
    The model has a Gaussian Activation layer where the probability
    of each input in each of the gaussian components is computed and
    used as location representation from which word distributions are
    learned.
    
    The learned word distributions can be used to detect local terms
    from a given region/location and also the learned Gaussians in
    the hidden layer are representing the dialect regions.
    """
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
        """
        build the network with 2d location as input,
        a bivariate Gaussian activated layer as hidden layer,
        a tanh layer as another hidden layer and a softmax
        probability distribution over vocabulary as output.
        
        Note that we didn't add regularization/dropout because the input 
        didn't have noisy features but nevertheless it might worth experimenting with.
        
        """
        
        self.X_sym = T.matrix()
        self.Y_sym = T.matrix()

        l_in = lasagne.layers.InputLayer(shape=(None, self.input_size),
                                         input_var=self.X_sym)


        
        logging.info('adding %d-comp bivariate gaussian layer...' %self.n_gaus_comp)
        l_gaus = BivariateGaussianLayer(l_in, num_units=self.n_gaus_comp, mus=self.mus, sigmas=self.sigmas, corxy=self.corxy)

        


        l_hid = lasagne.layers.DenseLayer(l_gaus, num_units=self.hid_size, 
                                          nonlinearity=lasagne.nonlinearities.rectify,
                                          W=lasagne.init.GlorotUniform())
        
        self.l_out = lasagne.layers.DenseLayer(l_hid, num_units=self.output_size,
                                          nonlinearity=lasagne.nonlinearities.softmax,
                                          W=lasagne.init.GlorotUniform())
        
        
        self.gaus_output = lasagne.layers.get_output(l_gaus, self.X_sym)
        self.eval_output = lasagne.layers.get_output(self.l_out, self.X_sym, deterministic=True)
        self.output = lasagne.layers.get_output(self.l_out, self.X_sym)
        loss = lasagne.objectives.categorical_crossentropy(self.output, self.Y_sym)
        loss = loss.mean()
        eval_loss = lasagne.objectives.categorical_crossentropy(self.eval_output, self.Y_sym)
        eval_loss = eval_loss.mean()

      
        parameters = lasagne.layers.get_all_params(self.l_out, trainable=True)
        updates = lasagne.updates.adamax(loss, parameters, learning_rate=2e-3, beta1=0.9, beta2=0.999, epsilon=1e-8)
        self.f_gaus = theano.function([self.X_sym], self.gaus_output, on_unused_input='warn')
        self.f_train = theano.function([self.X_sym, self.Y_sym], loss, updates=updates, on_unused_input='warn')
        self.f_val = theano.function([self.X_sym, self.Y_sym], eval_loss, on_unused_input='warn')
        self.f_predict_proba = theano.function([self.X_sym], self.eval_output, on_unused_input='warn') 
     

    def set_params(self, params):
        lasagne.layers.set_all_param_values(self.l_out, params)
        self.best_params = params

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
            
            vis_gaussians_during_training = False
            for step in range(self.n_epochs):
                if vis_gaussians_during_training:
                    #visualize learned Gaussian components in each 10 iterations (makes the training slower)
                    if step % 10 == 0:    
                        best_params = lasagne.layers.get_all_param_values(self.l_out)
                        visualise_gaussians(params=best_params, iter=step, output_type='png')
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
                        logging.info('first mu (%f,%f) first covar (%f, %f, %f)' %(best_params[0][0, 0], best_params[0][0, 1], softplus(best_params[1][0, 0]), softplus(best_params[1][0, 1]), softsign(best_params[2][0])))
                        logging.info('second mu (%f,%f) second covar (%f, %f, %f)' %(best_params[0][1, 0], best_params[0][1, 1], softplus(best_params[1][1, 0]), softplus(best_params[1][1, 1]), softsign(best_params[2][1])))
                    n_validation_down = 0
                else:
                    n_validation_down += 1
                    if n_validation_down > self.early_stopping_max_down:
                        logging.info('validation results went down. early stopping ...')
                        break
    
                logging.info('iter %d, train loss %f, dev loss %f, best dev loss %f, num_down %d' %(step, l_train, l_val, best_val_loss, n_validation_down))
            
            lasagne.layers.set_all_param_values(self.l_out, best_params)
            self.best_params = best_params
            #for debugging the output of gaussian layer
            #debug_gaussian(best_params)
            logging.info('dumping the model in %s' %model_file)
            with open(model_file, 'wb') as fout:
                pickle.dump(best_params, fout)        
        
        l_test = self.f_val(X_test, Y_test)
        perplexity_test = np.power(2, l_test)
        logging.info('test loss is %f and perplexity is %f' %(l_test, perplexity_test))
        l_dev = self.f_val(X_dev, Y_dev)
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
    normalize_words = kwargs.get('norm', False)


    vocab = None
    vocab_file = './dumps/na_vocab.pkl'
    if 'na' in dataset_name and path.exists(vocab_file):
        with open(vocab_file, 'rb') as fin:
            vocab = pickle.load(fin)
    extract_vocab = False
    norm = 'l1'
    idf = True
    if extract_vocab:
        norm = None
        idf = False
        
    dl = DataLoader(data_home=data_home, bucket_size=bucket_size, encoding=encoding, 
                    celebrity_threshold=celebrity_threshold, one_hot_labels=one_hot_label, 
                    mindf=mindf, maxdf=0.1, norm=norm, idf=idf, btf=True, tokenizer=None, 
                    subtf=True, stops=stop_words, token_pattern=r'(?u)(?<![@])#?\b\w\w+\b', vocab=vocab)
    logging.info('loading dataset...')
    dl.load_data()

    #load words that often start with uppercase (heuristic named entity detection)
    ne_file = './dumps/ne_' + dataset_name + '.json'
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
    printable_dialect_ranking = {d:rank[0:200] for d, rank in dialect_ranking.iteritems()}
    with open('./dumps/dialect_ranking_{}_hid{}_comp{}.json'.format(len(vocab), model.hid_size, model.n_gaus_comp) , 'w') as fout:
        json.dump(printable_dialect_ranking, fout, indent=4, sort_keys=True)
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
    
        
        
def city_dialect_words(model, vocab, filename='./city_ranking.txt'):
    #load named entities
    ne_file = './dumps/ne_' + dataset_name + '.json'
    with codecs.open(ne_file, 'r', encoding='utf-8') as fout:
        NEs = json.load(fout)
    NEs = set(NEs['nes'])
    
    k = 200
    with open('./data/cities.json', 'r') as fin:
        cities = json.load(fin)
    all_locs = np.array([[city['latitude'], city['longitude']] for city in cities]).astype('float32')
    all_probs = model.predict(all_locs)
    all_logprobs = np.log(all_probs)
    all_logprobs_mean = np.mean(all_logprobs, axis=0)
    city_dialectwords = defaultdict(list)
    
    cities = cities[0:200]
    for city in cities:
        name = city['city']
        lat, lon = city['latitude'], city['longitude']
        loc = np.array([[lat, lon]]).astype('float32')
        city_probs = model.predict(loc)
        city_logprobs = np.log(city_probs)
        normalized_city_logprobs = city_logprobs - all_logprobs_mean
        sorted_vocab_indices = np.argsort(normalized_city_logprobs)
        topwords = list(reversed(np.array(vocab)[sorted_vocab_indices][0].tolist()))[0:k]

        #check if a topword is a named entity add a star beside it
        dialect_words = []
        for topword in topwords:
            if topword in NEs:
                topword = "NE_" + topword
            dialect_words.append(topword)

        city_dialectwords[name] = dialect_words
        #write the city_dialectwords to file
        with codecs.open(filename, 'w', encoding='utf-8') as fout:
            json.dump(city_dialectwords, fout, indent=4, sort_keys=True)

           
        
        
    
      
    
def train(data, **kwargs):
    dropout_coef = kwargs.get('dropout_coef', 0.5)
    regul = kwargs.get('regul_coef', 1e-6)
    hid_size = kwargs.get('hidden_size', 500)
    autoencoder = kwargs.get('autoencoder', False)
    n_gaus_comp = kwargs.get('ncomp', 500)
    dataset_name = kwargs.get('dataset_name')
    kmeans_mu = kwargs.get('kmeans', True)
    nomdn = kwargs.get('nomdn', False)
    tune = kwargs.get('tune', False)
    loc_train, W_train, loc_dev, W_dev, loc_test, W_test, vocab = data
    input_size = loc_train.shape[1]
    output_size = W_train.shape[1]
    batch_size = kwargs.get('batch', 4000)
    vis_words = kwargs.get('map', True)
    vbi = kwargs.get('vbi', True)
    reload = kwargs.get('reload', False)
    epochs = kwargs.get('epochs', 1000)

    
    
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
                for i in range(mus.shape[0]):
                    logging.info('set all mus to the center of USA with a little noise')
                    mus[i, 0] = 39.5 + np.random.uniform(low=-3, high=+3)
                    mus[i, 1] = -98.35 + np.random.uniform(low=-3, high=+3)
        mus = mus.astype('float32')
        raw_stds = None
        raw_cors = None
    
    model = Loc2Lang(n_epochs=epochs, batch_size=batch_size, regul_coef=regul, 
                    input_size=input_size, output_size=output_size, hid_size=hid_size, 
                    drop_out=False, dropout_coef=dropout_coef, early_stopping_max_down=3, 
                    autoencoder=autoencoder, reload=reload, n_gaus_comp=n_gaus_comp, mus=mus, 
                    sigmas=raw_stds, corxy=raw_cors, nomdn=nomdn, dataset_name=dataset_name)
    #pdb.set_trace()
    perplexity_test, perplexity_dev = model.fit(loc_train, W_train, loc_dev, W_dev, loc_test, W_test)
    #model.fit(loc_train, loc_train, loc_dev, loc_dev, loc_test, loc_test)
    
    
    state_dialect_words(loc_train, vocab, model, N=10000 if dataset_name=='na' else 5000)
    
    #in case of tuning we don't want to visualize anything
    if tune:
        return perplexity_test, perplexity_dev
    
            
    filename = './dumps/local_words_{}_{}.txt'.format(str(W_train.shape), n_gaus_comp)
    city_dialect_words(model, vocab, filename=filename)
    
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
    if vis_words:
        map_words(coords, preds, vocab, map_dir='./maps/{}_voc{}_comp{}/'.format(dataset_name, W_train.shape[1], n_gaus_comp), dataset_name=dataset_name)
    if vbi: 
        #visualize the learned gaussians over the map
        visualise_gaussians(params=model.best_params, iter=None, output_type='pdf')      
    
def get_local_words(preds, vocab, NEs=[], k=50):
    """
    given the word probabilities over many coordinates,
    first normalize the probability of each word in different
    locations to get a probability distribution, then compute
    the entropy of the word's distribution over all coordinates
    and return the words that are low entropy and are not
    named entities.
    """
    #normalize the probabilites of each vocab using entropy
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
   
def map_words(coords, preds, vocab, map_dir, dataset_name):
    """
    given the coords distributed over the map and
    the unigram distribution over vocabulary pred,
    contourf the logprob of a word over the map
    with interpolation.
    """
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
    
            

    #if os.path.exists(map_dir):
    #    shutil.rmtree(map_dir)
    try:
        os.mkdir(map_dir)
    except:
        logging.info('map_dir %s exists or can not be created.')
    
    #pick some words to map including some known dialect words
    #some DARE words and some words that are not evenly distributed
    topk_words = []    
    for words in region_words.values():
        topk_words.extend(words)
    topk_words.extend(word_dialect.keys())
    dialect_words = ['hella', 'yall', 'jawn', 'paczki', 'euchre', 'brat', 'toboggan', 'brook', 'grinder', 'yinz', 'youze', 'yeen']
    topk_words.extend(dialect_words)
    custom_words = ['springfield', 'columbia', 'nigga', 'niqqa', 'bamma', 'cooter', 'britches', 'yapper', 'younguns', 'hotdish', 
                    'schnookered', 'bubbler', 'betcha', 'dontcha']
    topk_words.extend(custom_words)
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
            filename = '{}{}_{}.pdf'.format(map_dir, word.encode('utf-8'), grid_interpolation_method)
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
            del m


def visualise_gaussians(params=None, iter=None, output_type='pdf', **kwargs):
    """
    Visualize the bivariate Gaussians learned from the model over a map.
    params is the best parameters of NN model over development set.
    
    Note that the parameters are raw and the restrictions to put them in range
    are not yet applied and should be applied here (e.g. softsign, softplus).
    """
    mus, sigmas, corxys = params[0], params[1], params[2] 
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
        
        #note that corxy = 1.0 / (1 + np.abs(sigmaxy))
        Z = mlab.bivariate_normal(X, Y, sigmax=sigmax, sigmay=sigmay, mux=mux, muy=muy, sigmaxy=sigmaxy)

        #Z = maskoceans(X, Y, Z)
        
        #note that you can adjust levels if there are no contours or they are big
        #con = m.contour(X, Y, Z, levels=[0.01], linewidths=0.5, colors='darkorange', antialiased=True)
        con = m.contour(X, Y, Z, 0, linewidths=0.5, colors='darkorange', antialiased=True)
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
    filename = './maps/video/{}_gaus_{}_{}'.format(dataset_name, iter, output_type)
    plt.savefig(filename, frameon=False, dpi=200)
    plt.close()

def tune(data, dataset_name, args, num_iter=100):
    """
    Tune the hyper-parameters of the model.
    """
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
    parser.add_argument( '-batch','--batch', metavar='int',  help='SGD batch size',  type=int, default=500)
    parser.add_argument( '-hid','--hidden', metavar='int',  help='Hidden layer size after bigaus layer',  type=int, default=500)
    parser.add_argument( '-mindf','--mindf', metavar='int',  help='minimum document frequency in BoW',  type=int, default=10)
    parser.add_argument( '-d','--dir', metavar='str',  help='home directory',  type=str, default='./data')
    parser.add_argument( '-enc','--encoding', metavar='str',  help='Data Encoding (e.g. latin1, utf-8)',  type=str, default='utf-8')
    parser.add_argument( '-reg','--regularization', metavar='float',  help='regularization coefficient)',  type=float, default=1e-6)
    parser.add_argument( '-drop','--dropout', metavar='float',  help='dropout coef default 0.5',  type=float, default=0.5)
    parser.add_argument( '-map', '--map', action='store_true',  help='if true just draw maps from pre-trained model')
    parser.add_argument( '-tune', '--tune', action='store_true',  help='if true tune the hyper-parameters')  
    parser.add_argument( '-autoencoder', '--autoencoder', type=int,  help='the number of autoencoder steps before training', default=0)   
    parser.add_argument( '-ncomp', type=int,  help='the number of bivariate gaussians after the input layer', default=500) 
    parser.add_argument( '-vbi', '--vbi', type=str,  help='if exists load params from vbi file and visualize bivariate gaussians on a map', default=None)
    parser.add_argument( '-reload', action='store_true',  help='if true try to reload the parameters of network from pickle file')
    parser.add_argument( '-epochs', metavar='int',  help='maximum number of epochs for optimization',  type=int, default=1000) 
    args = parser.parse_args(argv)
    return args
if __name__ == '__main__':
    #THEANO_FLAGS='device=cpu'   nice -n 10 python loc2lang.py -d ~/datasets/na/processed_data/ -enc utf-8 -reg 0.0 -drop 0.0 -mindf 100 -hid 1000 -ncomp 500 -batch 5000
    if not path.exists("./dumps"):
        os.mkdir("./dumps")
    if not path.exists("./maps"):
        os.mkdir("./maps")
        os.mkdir("./maps/video")
    args = parse_args(sys.argv[1:])
    datadir = args.dir
    dataset_name = 'cmu' if 'cmu' in datadir else 'na'
    logging.info('dataset: %s' % dataset_name)
    data = load_data(data_home=args.dir, encoding=args.encoding, mindf=args.mindf, dataset_name=dataset_name)
    if args.tune:
        tune(data, dataset_name, args, num_iter=100)
    else:
        train(data, regul_coef=args.regularization, dropout_coef=args.dropout, 
              hidden_size=args.hidden, autoencoder=args.autoencoder, ncomp=args.ncomp, dataset_name=dataset_name, batch=args.batch, reload=args.reload, epochs=args.epochs)
