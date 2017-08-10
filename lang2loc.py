'''
Created on 21 Feb 2017
Given a training set of lat/lon as input and probability distribution over words as output,
train a model that can predict words based on location.
then try to visualise borders and regions (e.g. try many lat/lon as input and get the probability of word yinz
in the output and visualise that).
@author: af
'''
import argparse
from collections import OrderedDict
import logging
from os import path
import pickle
import random
import sys

from haversine import haversine
import lasagne
from lasagne.regularization import l2, l1
from sklearn.cluster import MiniBatchKMeans
import theano

from data import DataLoader
from lasagne_layers import SparseInputDenseLayer
import lasagne_layers
import numpy as np
import scipy as sp
import theano.sparse as S
import theano.tensor as T
from utils import stop_words
import utils


logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

np.random.seed(77)
random.seed(77)



def geo_latlon_eval(U_eval, userLocation, latlon_pred, contour_error_on_map=False):
    distances = []
    real_latlons = []
    for i in range(0, len(U_eval)):
        user = U_eval[i]
        location = userLocation[user].split(',')
        lat, lon = float(location[0]), float(location[1])
        real_latlons.append([lat, lon])
        lat_pred, lon_pred = latlon_pred[i]
        distance = haversine((lat, lon), (lat_pred, lon_pred))
        distances.append(distance)
    acc_at_161 = 100 * len([d for d in distances if d < 161]) / float(len(distances))
    logging.info( "Mean: " + str(int(np.mean(distances))) + " Median: " + str(int(np.median(distances))) + " Acc@161: " + str(int(acc_at_161)))
    if contour_error_on_map:
        coordinates = np.array(real_latlons)
        utils.contour(coordinates, distances, filename='distance_contour_' + str(np.median(distances)))
    return np.mean(distances), np.median(distances), acc_at_161

class NNModel_lang2loc():
    def __init__(self, 
                 n_epochs=10, 
                 batch_size=1000, 
                 regul_coef=1e-6,
                 input_size=None,
                 output_size = None, 
                 hid_size=100, 
                 drop_out=False, 
                 dropout_coef=0.5,
                 early_stopping_max_down=10,
                 dtype='float32',
                 autoencoder=100,
                 input_sparse=False,
                 reload=False,
                 ncomp=100,
                 sqerror=False,
                 dataset_name=''):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.regul_coef = regul_coef
        self.hid_size = hid_size
        self.drop_out = drop_out
        self.dropout_coef = dropout_coef
        self.early_stopping_max_down = early_stopping_max_down
        self.dtype = dtype
        self.input_size = input_size
        self.output_size = output_size
        self.autoencoder = autoencoder
        self.sparse = input_sparse
        self.reload = reload
        self.n_bigaus_comp = ncomp
        self.sqerror = sqerror
        self.dataset_name = dataset_name
        logging.info('building nn model with %d hidden size, %d bivariate gaussian components and %d output size' % (self.hid_size, self.n_bigaus_comp, self.output_size) )
        if self.sqerror:
            self.build_squarederror_regression()
        else:
            self.build()
            
        


    def unpack_params(self, output, n_comp):
        """
        Given the output of a tanh dense layer with 6 * n_comp size for a batch of input
        reshape the output and extract the mus, sigmas, corxy and pis of each sample.
        Apply restrictions on the value of each parameter of mixture of Gaussians so
        that they fall within the allowed range:
        mus are latitude/longitude and for US latitude should be between (0, 90)
        and longitude should be between (-180,0).
        sigmas should be positive (0, +inf) apply softplus and because we want them to be
        large enough multiply them by 10 which is an empirically chosen multiplier.
        pis should be positive and sum to 1 apply softmax
        corxy should be in (-1, 1): apply softwign 
        """
        output = output.reshape(shape=(-1, 6, n_comp))
        mus = output[:, 0:2, :]
        mus = np.array([90, -180])[np.newaxis, :, np.newaxis] * T.nnet.sigmoid(mus)
        sigmas = output[:, 2:4, :]
        corxy = output[:, 4, :]
        pis = output[:, 5, :]
        #sigmas should be positive (0, +inf)
        sigmas = 10 * T.nnet.softplus(sigmas)
        #sigmas = T.exp(sigmas)
        #pis should sum to 1 for each input
        #clip each pi based on https://github.com/axelbrando/Mixture-Density-Networks-for-distribution-and-uncertainty-estimation/blob/master/MDN-2D-Regression.ipynb
        #pis = T.clip(pis, 1e-8, 100)
        pis = T.nnet.softmax(pis)
        #cor(x, y) should be between (-1, 1)
        corxy = T.nnet.nnet.softsign(corxy)
        #corxy = T.tanh(corxy)
        return mus, sigmas, corxy, pis

 
 


    def nll_loss(self, mus, sigmas, corxy, pis, y_true):
        """
        negative log likelihood loss of a 2d y_true coordinate in
        each of the Gaussians with parameters mus, sigmas, corxy, pis.
        Note that the mus, sigmas and corxy are shared between all samples
        and only pis are different for each sample.
        
        The formula for negative log likelihood is :
        \mathcal{L}(y \vert x) = - \log\bigg\{\sum_{k=1}^K \pi_k(x)  \mathcal{N}\big(y \vert \mu_k(x), \Sigma_k(x)\big)\bigg\}
        
        The size of pis is n_batch x n_components,
        the size of mus is n_batch x n_components x 2,
        the size of sigmas is n_batch x n_components x 2 and
        the size of corxy is n_batch x n_components.
        
        The size of y_true is batch_size x 2.
        """

        Y = y_true[:, :, np.newaxis]
        diff = Y - mus
        diffprod = T.prod(diff, axis=-2)
        sigmainvs = T.inv(sigmas)
        sigmainvprods = sigmainvs[:,0, :] * sigmainvs[:,1, :]
        sigmas2 = sigmas ** 2
        corxy2 = corxy **2
        diff2 = diff ** 2
        diffsigma = diff2 * T.inv(sigmas2)
        diffsigmanorm = T.sum(diffsigma, axis=-2)
        z = diffsigmanorm - 2 * corxy * diffprod * sigmainvprods
        oneminuscorxy2inv = T.inv(1.0 - corxy2)
        '''
        expterm = T.exp(-0.5 * z * oneminuscorxy2inv)
        probs = (0.5 / np.pi) * sigmainvprods * T.sqrt(oneminuscorxy2inv) * expterm
        loss = - T.log(T.sum(pis * probs, axis=1))
        loss = T.mean(loss)
        '''
        #logsumexp trick
        exponent = -0.5 * z * oneminuscorxy2inv
        #normalizer = (0.5 / np.pi) * sigmainvprods * T.sqrt(oneminuscorxy2inv)
        #when something is a * exp(x) = exp(x + loga)
        new_exponent = exponent + T.log(0.5 / np.pi) + T.log(sigmainvprods) + T.log(T.sqrt(oneminuscorxy2inv)) + T.log(pis)
        max_exponent = T.max(new_exponent ,axis=1, keepdims=True)
        mod_exponent = new_exponent - max_exponent
        gauss_mix = T.sum(T.exp(mod_exponent),axis=1)
        log_gauss = max_exponent + T.log(gauss_mix)
        loss = -T.mean(log_gauss)
        
        return loss
    
    def pred(self, mus, sigmas, corxy, pis, prediction_method='mixture'):
        """
        Given a mixture of Gaussians infer a mu that maximizes the mixture.
        There are two modes:
        If prediction_method==mixture then predict one of the mus that maximizes
        \mathcal{P}(\boldsymbol{x}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\boldsymbol{x} \vert \boldsymbol{\mu_k}, \Sigma_k)
        
        If prediction_method==pi return the mu that has the largest pi.
        """
        if prediction_method == 'mixture':
            #logging.info('predicting the best mixture mus')
            X = mus[:, :, :, np.newaxis]
            musex = mus[:, :, np.newaxis, :]
            sigmasex = sigmas[:, :, :, np.newaxis]
            corxysex = corxy[:, :, np.newaxis]
            diff = X - musex
            diffprod = np.prod(diff, axis=-3)
            sigmainvs = 1.0 / sigmasex
            sigmainvprods = sigmainvs[:,0, :, :] * sigmainvs[:,1, :, :]
            sigmas2 = sigmas ** 2
            corxy2 = corxysex **2
            diff2 = diff ** 2
            diffsigma = diff2 / sigmas2[:, :, :, np.newaxis]
            diffsigmanorm = np.sum(diffsigma, axis=-3)
            z = diffsigmanorm - 2 * corxysex * diffprod * sigmainvprods
            oneminuscorxy2inv = 1.0 / (1.0 - corxy2)
            term = -0.5 * z * oneminuscorxy2inv
            expterm = np.exp(term)
            probs = (0.5 / np.pi) * sigmainvprods * np.sqrt(oneminuscorxy2inv) * expterm
            piprob = pis[:, :, np.newaxis] * probs
            piprobsum = np.sum(piprob, axis=-2)
            preds = np.argmax(piprobsum, axis=1)
            selected_mus = mus[np.arange(mus.shape[0]),:,preds]
            
            #selected_sigmas = sigmas[np.arange(sigmas.shape[0]),preds,:]
            #selected_corxy = corxy[np.arange(corxy.shape[0]),preds]
            #selected_pis = pis[np.arange(pis.shape[0]),preds]        
            return selected_mus
        elif prediction_method == 'pi':
            #logging.info(sigmas[0])
            #logging.info(pis[0])
            #logging.info(corxy[0])
            
            logging.info('only pis are used for prediction')
            preds = np.argmax(pis, axis=1)
            selected_mus = mus[np.arange(mus.shape[0]), :, preds]
            #selected_sigmas = sigmas[np.arange(sigmas.shape[0]), :, preds]
            #selected_corxy = corxy[np.arange(corxy.shape[0]),preds]
            #selected_pis = pis[np.arange(pis.shape[0]),preds]        
            return selected_mus
        elif prediction_method == 'mixture':
            logging.info('not implemented!')
 
  
            
    def get_symb_mus(self, mus, sigmas, corxy, pis, prediction_method="pi"):
        """
        Can be used to train an autoencoder that given location
        trains a mixture density layer and then outputs the same
        location
        symbolycally predict the mu that maximizes the mixture model
        either based on mixture probability of the component
        with highest pi, see pred_sharedparams
        """
        if prediction_method == "mixture":
            #sigmainvs = 1.0 / sigmas
            #sigmainvprods = sigmainvs[:,:, 0] * sigmainvs[:,:, 1]
            #sigmas2 = sigmas ** 2
            #corxy2 = corxy **2
            #diff2 = diff ** 2
            #diffsigma = diff2 / sigmas2
            #diffsigmanorm = np.sum(diffsigma, axis=-1)
            #z = diffsigmanorm - 2 * corxy * diffprod * sigmainvprods
            #oneminuscorxy2inv = 1.0 / (1.0 - corxy2)
            #expterm = np.exp(-0.5 * z * oneminuscorxy2inv)
            #expterm = 1.0
            #probs = (0.5 / np.pi) * sigmainvprods * T.sqrt(oneminuscorxy2inv) * expterm
            #probs = pis * probs
            logging.fatal("not implemented!")
            sys.exit()
        elif prediction_method == "pi":
            preds = T.argmax(pis, axis=1)
            selected_mus = mus[T.arange(mus.shape[0]),preds,:]
            return selected_mus


    
    def build(self):

        self.X_sym = S.csr_matrix(name='inputs', dtype=self.dtype)
        self.Y_sym = T.matrix(name='y_true', dtype=self.dtype)

        l_in_text = lasagne.layers.InputLayer(shape=(None, self.input_size),
                                         input_var=self.X_sym)


        if self.drop_out and self.dropout_coef > 0:
            l_in_text = lasagne_layers.SparseInputDropoutLayer(l_in_text, p=self.dropout_coef)

        
        l_hid_text = SparseInputDenseLayer(l_in_text, num_units=self.hid_size, 
                                      nonlinearity=lasagne.nonlinearities.tanh,
                                      W=lasagne.init.GlorotUniform())

        #if self.drop_out and self.dropout_coef > 0:
        #    l_hid_text = lasagne.layers.dropout(l_hid_text, p=self.dropout_coef)


        self.l_out_gaus = lasagne.layers.DenseLayer(l_hid_text, num_units=self.n_bigaus_comp * 6,
                                          nonlinearity=lasagne.nonlinearities.linear,
                                          W=lasagne.init.GlorotUniform())
            
        
        output = lasagne.layers.get_output(self.l_out_gaus, self.X_sym)
        mus, sigmas, corxy, pis = self.unpack_params(output, n_comp=self.n_bigaus_comp)
        loss = self.nll_loss(mus, sigmas, corxy, pis, self.Y_sym)
        #we can add an autoencoder loss as well
        #sq_error_coef = 0.01
        #predicted_mu = self.get_symb_mus(mus, sigmas, corxy, pis)
        #loss += lasagne.objectives.squared_error(predicted_mu, self.Y_sym).mean() * sq_error_coef
  
        if self.regul_coef:
            l1_share_out = 0.5
            l1_share_hid = 0.5
            regul_coef_out, regul_coef_hid = self.regul_coef, self.regul_coef
            logging.info('regul coefficient for output and hidden lasagne_layers is ' + str(self.regul_coef))
            #l1_penalty = lasagne.regularization.regularize_layer_params(self.l_out_gaus, l1) * regul_coef_out * l1_share_out
            #l2_penalty = lasagne.regularization.regularize_layer_params(self.l_out_gaus, l2) * regul_coef_out * (1-l1_share_out)
            l1_penalty = lasagne.regularization.regularize_layer_params(l_hid_text, l1) * regul_coef_hid * l1_share_hid
            l2_penalty = lasagne.regularization.regularize_layer_params(l_hid_text, l2) * regul_coef_hid * (1-l1_share_hid)

            loss += l1_penalty + l2_penalty


        
        parameters = lasagne.layers.get_all_params(self.l_out_gaus, trainable=True)
        updates = lasagne.updates.adam(loss, parameters, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8)
        self.f_train = theano.function([self.X_sym, self.Y_sym], loss, updates=updates, on_unused_input='warn')
        self.f_val = theano.function([self.X_sym, self.Y_sym], loss, on_unused_input='warn')
        self.f_predict = theano.function([self.X_sym], [mus, sigmas, corxy, pis], on_unused_input='warn')


     

    def build_squarederror_regression(self):
        """
        used in case we want to build a regression model that predicts 2d location coordinates
        from input text.
        """
        self.X_sym = S.csr_matrix(name='inputs', dtype=self.dtype)
        self.Y_sym = T.matrix(name='y_true', dtype=self.dtype)
        self.X_autoencoder_sym = T.matrix(name='x_autoencoder', dtype=self.dtype)
        self.Y_autoencoder_sym = T.matrix(name='y_autoencoder', dtype=self.dtype)
        

        l_in_text = lasagne.layers.InputLayer(shape=(None, self.input_size),
                                         input_var=self.X_sym)
     
        

        if self.drop_out and self.dropout_coef > 0:
            l_in_text = lasagne_layers.SparseInputDropoutLayer(l_in_text, p=self.dropout_coef)

       
        
        l_hid_text = SparseInputDenseLayer(l_in_text, num_units=self.hid_size, 
                                      nonlinearity=lasagne.nonlinearities.tanh,
                                      W=lasagne.init.GlorotUniform())

        if self.drop_out and self.dropout_coef > 0:
            l_hid_text = lasagne.layers.dropout(l_hid_text, p=self.dropout_coef)

 
        self.l_out = lasagne.layers.DenseLayer(l_hid_text, num_units=2,
                                               nonlinearity=lasagne.nonlinearities.linear,
                                               W=lasagne.init.GlorotUniform())
            
            

        output = lasagne.layers.get_output(self.l_out, self.X_sym)
        loss = lasagne.objectives.squared_error(output, self.Y_sym).mean() 
        output_eval = lasagne.layers.get_output(self.l_out, self.X_sym, deterministic=True)


            
  
        if self.regul_coef:
            l1_share_out = 0.5
            l1_share_hid = 0.5
            regul_coef_out, regul_coef_hid = self.regul_coef, self.regul_coef
            logging.info('regul coefficient for output and hidden lasagne_layers is ' + str(self.regul_coef))
            #l1_penalty = lasagne.regularization.regularize_layer_params(self.l_out, l1) * regul_coef_out * l1_share_out
            #l2_penalty = lasagne.regularization.regularize_layer_params(self.l_out, l2) * regul_coef_out * (1-l1_share_out)
            l1_penalty = lasagne.regularization.regularize_layer_params(l_hid_text, l1) * regul_coef_hid * l1_share_hid
            l2_penalty = lasagne.regularization.regularize_layer_params(l_hid_text, l2) * regul_coef_hid * (1-l1_share_hid)


            loss = loss + l1_penalty + l2_penalty


        
        parameters = lasagne.layers.get_all_params(self.l_out, trainable=True)
        updates = lasagne.updates.adam(loss, parameters, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8)
        self.f_train = theano.function([self.X_sym, self.Y_sym], loss, updates=updates, on_unused_input='warn')
        self.f_val = theano.function([self.X_sym, self.Y_sym], loss, on_unused_input='warn')
        self.f_predict = theano.function([self.X_sym], output_eval, on_unused_input='warn')
        
    def set_params(self, params):
        lasagne.layers.set_all_param_values(self.l_out_gaus, params)

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
    
    def fit(self, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, userLocation):
        model_file = './data/lang2loc_%s_hid%d_gaus%d.pkl' %(self.dataset_name, self.hid_size, self.n_bigaus_comp)
        if self.reload:
            if path.exists(model_file):
                logging.info('loading the model from %s' %model_file)
                with open(model_file, 'rb') as fin:
                    params = pickle.load(fin)
                self.set_params(params)
                return

               
        logging.info('training with %d n_epochs and  %d batch_size' %(self.n_epochs, self.batch_size))
        best_params = None
        best_val_loss = sys.maxint
        n_validation_down = 0
        
        for step in range(self.n_epochs):
            l_trains = []
            for batch in self.iterate_minibatches(X_train, Y_train, self.batch_size, shuffle=True):
                x_batch, y_batch = batch
                l_train = self.f_train(x_batch, y_batch)
                l_trains.append(l_train)
            l_train = np.mean(l_trains)
            #because dev set is big we can't predict in a row because of memory size
            if self.dataset_name == "na":
                l_vals = []
                for batch in self.iterate_minibatches(X_dev, Y_dev, self.batch_size, shuffle=False):
                    x_batch, y_batch = batch
                    l_val = self.f_val(x_batch, y_batch)
                    l_vals.append(l_val)
                l_val = np.mean(l_vals)
            else:
                l_val = self.f_val(X_dev, Y_dev)
            #preds = self.predict(X_dev)

            if l_val < best_val_loss:
                best_val_loss = l_val
                if self.sqerror:
                    best_params = lasagne.layers.get_all_param_values(self.l_out)
                else:
                    best_params = lasagne.layers.get_all_param_values(self.l_out_gaus)
                n_validation_down = 0
            else:
                n_validation_down += 1
                if n_validation_down > self.early_stopping_max_down:
                    logging.info('validation results went down. early stopping ...')
                    break
            logging.info('iter %d, train loss %f, dev loss %f, best dev loss %f, num_down %d' %(step, l_train, l_val, best_val_loss, n_validation_down))
        if self.sqerror:
            lasagne.layers.set_all_param_values(self.l_out, best_params)
        else:
            lasagne.layers.set_all_param_values(self.l_out_gaus, best_params)

                
    def predict(self, X):
        mus_eval, sigmas_eval, corxy_eval, pis_eval = self.f_predict(X)
        selected_mus = self.pred(mus_eval, sigmas_eval, corxy_eval, pis_eval)
        return selected_mus       

 
    def predict_regression(self, X):
        output = self.f_predict(X)
        return output


    
          
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
    city_stops = kwargs.get('city_stops', False)

    dl = DataLoader(data_home=data_home, bucket_size=bucket_size, encoding=encoding, 
                    celebrity_threshold=celebrity_threshold, one_hot_labels=one_hot_label, 
                    mindf=mindf, maxdf=0.1, norm='l2', idf=True, btf=True, tokenizer=None, subtf=True, stops=stop_words, token_pattern=r'(?u)(?<![@])\b\w+\b')
    dl.load_data()
    
    dl.tfidf()
    U_test = dl.df_test.index.tolist()
    U_dev = dl.df_dev.index.tolist()
    U_train = dl.df_train.index.tolist()  
    X_train = dl.X_train.astype(dtype)
    X_dev = dl.X_dev.astype(dtype)
    X_test = dl.X_test.astype(dtype)
    classLatMedian, classLonMedian = None, None
    loc_train = np.array([[a[0], a[1]] for a in dl.df_train[['lat', 'lon']].values.tolist()], dtype=dtype)
    loc_dev = np.array([[a[0], a[1]] for a in dl.df_dev[['lat', 'lon']].values.tolist()], dtype=dtype)
    loc_test = np.array([[a[0], a[1]] for a in dl.df_test[['lat', 'lon']].values.tolist()], dtype=dtype)
    Y_train = loc_train
    Y_dev = loc_dev
    Y_test = loc_test
    P_test = [str(a[0]) + ',' + str(a[1]) for a in dl.df_test[['lat', 'lon']].values.tolist()]
    P_train = [str(a[0]) + ',' + str(a[1]) for a in dl.df_train[['lat', 'lon']].values.tolist()]
    P_dev = [str(a[0]) + ',' + str(a[1]) for a in dl.df_dev[['lat', 'lon']].values.tolist()]
    userLocation = {}
    for i, u in enumerate(U_train):
        userLocation[u] = P_train[i]
    for i, u in enumerate(U_test):
        userLocation[u] = P_test[i]
    for i, u in enumerate(U_dev):
        userLocation[u] = P_dev[i]
    
    data = (X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation, loc_train)
    return data

def load_toy_data(n_samples=1000, dtype='float32'):
    print('creating Melbourne toy dataset as an inverse problem.')
    print('There are two (if not more) Melbournes, one in Australia and one in Florida, USA')
    mlb_fl_latlon_mean = np.array((28.0836, -80.6081))
    mlb_au_latlon_mean = np.array((-37.8136, 144.9631))
    cov=np.array([[1, 0],[0, 1]])
    #create bivariate gaussians to sample from the means (with variances 1, 1 and correlation 0) Melb, Au samples are two times of Melb, FL
    mlb_fl_samples = np.random.multivariate_normal(mean=mlb_fl_latlon_mean, cov=cov, size=n_samples).astype(dtype)
    mlb_au_samples = np.random.multivariate_normal(mean=mlb_au_latlon_mean, cov=cov, size=n_samples * 2).astype(dtype)
    
    #plt.scatter(mlb_fl_samples[:, 0], mlb_fl_samples[:, 1], c='blue', s=1)
    #plt.scatter(mlb_au_samples[:, 0], mlb_au_samples[:, 1], c='red', s=1)
    #plt.show()

    X = sp.sparse.csr_matrix(np.random.uniform(-0.1, 0.1, size=(n_samples * 3, 2) ) + np.array([1, 0])).astype(dtype)
    
    Y = np.vstack((mlb_fl_samples, mlb_au_samples))  
    #shuffle X and Y
    indices = np.arange(n_samples * 3)
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    n_train_samples = 2 * n_samples
    n_dev_samples = n_samples / 2
    n_test_samples = 3 * n_samples - n_train_samples - n_dev_samples  
    X_train = X[0:n_train_samples, :]
    X_dev = X[n_train_samples:n_train_samples + n_dev_samples, :]
    X_test = X[n_train_samples + n_dev_samples:n_train_samples + n_dev_samples + n_test_samples, :]
    Y_train = Y[0:n_train_samples, :]
    Y_dev = Y[n_train_samples:n_train_samples + n_dev_samples, :]
    Y_test = Y[n_train_samples + n_dev_samples:n_train_samples + n_dev_samples + n_test_samples, :]
    U_train = [i for i in range(n_train_samples)]
    U_dev = [i for i in range(n_train_samples, n_train_samples + n_dev_samples)]
    U_test = [i for i in range(n_train_samples + n_dev_samples, n_train_samples + n_dev_samples + n_test_samples)]
    userLocation = {}
    for i in range(0, 3 * n_samples):
        lat, lon = Y[i, :]
        userLocation[i] = str(lat) + ',' + str(lon)
    data =  (X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, None, None, userLocation, None)
    return data
    
    
    
def train(data, **kwargs):
    np.random.seed(77)
    dropout_coef = kwargs.get('dropout_coef', 0.5)
    regul = kwargs.get('regul_coef', 1e-6)
    hid_size = kwargs.get('hidden_size', 200)
    autoencoder = kwargs.get('autoencoder', False)
    ncomp = kwargs.get('ncomp', 100)
    dataset_name = kwargs.get('dataset_name')
    batch_size = kwargs.get('batch_size', 200 if dataset_name=='cmu' else 1000)
    sqerror = kwargs.get('sqerror', False)
    X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation, loc_train = data
    input_size = X_train.shape[1]
    output_size = Y_train.shape[1] if len(Y_train.shape) == 2 else np.max(Y_train) + 1
    logging.info('batch size %d' % batch_size)
    max_down = 20 if dataset_name == 'cmu' else 5 
    
    
    model = NNModel_lang2loc(n_epochs=10000, batch_size=batch_size, regul_coef=regul, 
                    input_size=input_size, output_size=output_size, hid_size=hid_size, 
                    drop_out=True, dropout_coef=dropout_coef, early_stopping_max_down=max_down, 
                    input_sparse=True, reload=False, ncomp=ncomp, autoencoder=autoencoder, sqerror=sqerror, dataset_name=dataset_name)

    model.fit(X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, userLocation)
    #save some space before prediction by deleting X_train (which is big)
    del X_train
    del Y_train
    if model.sqerror:
        latlon_pred = model.predict_regression(X_dev)
    else:
        if dataset_name == 'cmu':
            latlon_pred = model.predict(X_dev)
        else:
            latlon_preds = []
            for batch in model.iterate_minibatches(X_dev, X_dev, model.batch_size, shuffle=False):
                x_batch, x_batch = batch
                latlon_pred = model.predict(x_batch)
                latlon_preds.append(latlon_pred)
            latlon_pred = np.vstack(tuple(latlon_preds))
    logging.info('dev results:')
    mean_dev , median_dev, acc_dev = geo_latlon_eval(U_dev, userLocation, latlon_pred, contour_error_on_map=False)

    logging.info('test results:')
    if model.sqerror:
        latlon_pred = model.predict_regression(X_test)
    else:
        if dataset_name == 'cmu':
            latlon_pred = model.predict(X_test)
        else:
            latlon_preds = []
            for batch in model.iterate_minibatches(X_test, X_test, model.batch_size, shuffle=False):
                x_batch, x_batch = batch
                latlon_pred = model.predict(x_batch)
                latlon_preds.append(latlon_pred)
            latlon_pred = np.vstack(tuple(latlon_preds))
    mean_test , median_test, acc_test = geo_latlon_eval(U_test, userLocation, latlon_pred)
    return mean_test, median_test, acc_test, mean_dev, median_dev, acc_dev
    #latlon_pred = model.predict(X_test)
    #geo_latlon_eval(U_test, userLocation, latlon_pred)

def tune(data, dataset_name, args, num_iter=100):
    logging.info('tuning over %s' %dataset_name)
    param_scores = []
    random.seed()
    ncomps = [100, 300, 900]
    hidden_sizes = [100, 300, 900]
    regul_coefs = [0, 1e-5]
    drop_out_ceofs = [0, 0.5]
    if args.sqerror: ncomps = [0]
    for ncomp in ncomps:
        for hidden_size in hidden_sizes:
            for regul_coef in regul_coefs:
                for drop_out_ceof in drop_out_ceofs:                    
                    np.random.seed(77)
                    logging.info('regul %f drop %f hidden %d ncomp %d' %(regul_coef, drop_out_ceof, hidden_size, ncomp))
                    mean_test, median_test, acc_test, mean_dev, median_dev, acc_dev = train(data, regul_coef=regul_coef, dropout_coef=drop_out_ceof, 
                          hidden_size=hidden_size, ncomp=ncomp, dataset_name=dataset_name, sqerror=args.sqerror)
                    scores = OrderedDict()
                    scores['mean_dev'], scores['median_dev'], scores['acc_dev'] = mean_dev, median_dev, acc_dev
                    scores['mean_test'], scores['median_test'], scores['acc_test'] = mean_test, median_test, acc_test
                    params = OrderedDict()
                    params['regul'], params['dropout'], params['hidden'], params['ncomp'] = regul_coef, drop_out_ceof, hidden_size, ncomp
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
    parser.add_argument('-i','--dataset', metavar='str', help='dataset for dialectology', type=str, default='na')
    parser.add_argument('-bucket','--bucket', metavar='int', help='discretisation bucket size', type=int, default=300)
    parser.add_argument('-batch','--batch', metavar='int', help='SGD batch size', type=int, default=0)
    parser.add_argument('-hid','--hidden', metavar='int', help='Hidden layer size', type=int, default=500)
    parser.add_argument('-mindf','--mindf', metavar='int', help='minimum document frequency in BoW', type=int, default=10)
    parser.add_argument('-d','--dir', metavar='str', help='home directory', type=str, default='./data')
    parser.add_argument('-enc','--encoding', metavar='str', help='Data Encoding (e.g. latin1, utf-8)', type=str, default='utf-8')
    parser.add_argument('-reg','--regularization', metavar='float', help='regularization coefficient)', type=float, default=1e-6)
    parser.add_argument('-drop','--dropout', metavar='float', help='dropout coef default 0.5', type=float, default=0.5)
    parser.add_argument('-cel','--celebrity', metavar='int', help='celebrity threshold', type=int, default=10)
    parser.add_argument('-conv', '--convolution', action='store_true', help='if true do convolution')
    parser.add_argument('-map', '--map', action='store_true', help='if true just draw maps from pre-trained model') 
    parser.add_argument('-sqerror', '--sqerror', action='store_true', help='if exists use squared error regression instead of gaussian mixture model') 
    parser.add_argument('-autoencoder', '--autoencoder', type=int, help='if not zero pre-trains the model with input lat/lon and output lat/lon for n steps', default=0) 
    parser.add_argument('-grid', '--grid', action='store_true', help='if exists transforms the input from lat/lon to distance from grids on map') 
    parser.add_argument('-rbf', '--rbf', action='store_true', help='if exists transforms the input from lat/lon to rbf probabilities and learns centers and sigmas as well.') 
    parser.add_argument('-ncomp', '--ncomp', type=int, help='the number of bivariate gaussians whose parameters are going to be learned.', default=100) 
    parser.add_argument('-toy', action='store_true', help='if exists use the toy dataset instead of geolocation datasets.')
    parser.add_argument('-tune', action='store_true', help='if exists tune hyperparameters')
    parser.add_argument('-m', '--message', type=str) 
    
    args = parser.parse_args(argv)
    return args

if __name__ == '__main__':
    #THEANO_FLAGS='device=cuda2' nice -n 10 python lang2loc.py -d ~/datasets/cmu/processed_data/ -enc latin1 -reg 0 -drop 0.5 -mindf 10 -hid 100 -ncomp 100
    #THEANO_FLAGS='device=cpu' nice -n 10 python lang2loc.py -d ~/datasets/na/processed_data/ -enc utf-8 -reg 1e-5 -drop 0.0 -mindf 10 -hid 300 -ncomp 100 
    args = parse_args(sys.argv[1:])
    datadir = args.dir
    dataset_name = datadir.split('/')[-3]
    logging.info('dataset: %s' % dataset_name)
    
    if args.toy:
        logging.info('toy dataset is being used.')
        data = load_toy_data()
    else:
        data = load_data(data_home=args.dir, encoding=args.encoding, mindf=args.mindf, grid=args.grid, dataset_name=dataset_name)
    
    if args.tune:
        tune(data, dataset_name, args)
    else:
        if not args.batch:
            batch_size = 200 if dataset_name=='cmu' else 1000
        else:
            batch_size = args.batch
        train(data, regul_coef=args.regularization, dropout_coef=args.dropout, 
              hidden_size=args.hidden, autoencoder=args.autoencoder, grid=args.grid, rbf=args.rbf, ncomp=args.ncomp, dataset_name=dataset_name, sqerror=args.sqerror, batch_size=batch_size)