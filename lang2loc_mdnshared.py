'''
Created on 21 Feb 2017
Given a training set of lat/lon as input and probability distribution over words as output,
train a model that can predict words based on location.
then try to visualise borders and regions (e.g. try many lat/lon as input and get the probability of word yinz
in the output and visualize it).
@author: af
'''

import argparse
from collections import OrderedDict
import logging
from os import path
import pdb
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

def softplus(x):
    return np.log(np.exp(x) + 1)

def softsign(x):
    return x / (1 + np.abs(x))



def geo_eval(y_true, y_pred, U_eval, classLatMedian, classLonMedian, userLocation):
    assert len(y_pred) == len(U_eval), "#preds: %d, #users: %d" % (len(y_pred), len(U_eval))
    distances = []
    for i in range(0, len(y_pred)):
        user = U_eval[i]
        location = userLocation[user].split(',')
        lat, lon = float(location[0]), float(location[1])
        prediction = str(y_pred[i])
        lat_pred, lon_pred = classLatMedian[prediction], classLonMedian[prediction]  
        distance = haversine((lat, lon), (lat_pred, lon_pred))
        distances.append(distance)

    acc_at_161 = 100 * len([d for d in distances if d < 161]) / float(len(distances))
    logging.info("Mean %f Median %f acc@161 %f" % (np.mean(distances), np.median(distances), acc_at_161))
    return np.mean(distances), np.median(distances), acc_at_161





def geo_latlon_eval(U_eval, userLocation, latlon_pred, contour_error_on_map=False, use_cluster_median=False, error_analysis=False):
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
    logging.info("Mean: " + str(int(np.mean(distances))) + " Median: " + str(int(np.median(distances))) + " Acc@161: " + str(int(acc_at_161)))
    if contour_error_on_map:
        coordinates = np.array(real_latlons)
        utils.contour(coordinates, distances, filename='distance_contour_' + str(np.median(distances)))
    if error_analysis:
        top_error_indices = list(reversed(np.argsort(distances).tolist()))[0:20]
        for _i in top_error_indices:
            u = U_eval[_i]
            location = userLocation[user]
            pred = str(latlon_pred[_i])
            logging.debug('user %s location %s pred %s' % (u, location, pred))
            logging.debug(dl.df_dev.text.values[_i])
            
        
    return np.mean(distances), np.median(distances), acc_at_161

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
        # rowvar should be False so that each column is considered a variable (not each row)
        covmat = np.cov(samples, rowvar=False)
        if samples.shape[0] == 1:
            # only one sample in the cluster
            stdlatlat = 1
            stdlonlon = 1
            covlatlon = 0
        else: 
            stdlatlat = np.sqrt(covmat[0, 0])
            stdlonlon = np.sqrt(covmat[1, 1])
            covlatlon = covmat[0, 1]
            if np.isnan(stdlatlat) or np.isnan(stdlonlon) or np.isnan(covlatlon):
                stdlatlat = 1
                stdlonlon = 1
                covlatlon = 0
            corlatlon = covlatlon / (stdlatlat * stdlonlon)
        
        increase_sigmas = False
        if increase_sigmas:
            stdlatlat *= 10
            stdlonlon *= 10
            corlatlon /= 100.0
           
        if raw:
            # softplus will be applied on sigmas so now apply the reverse so that they become sigmas in neural network
            sigmas[i, 0] = np.log(np.exp(stdlatlat) - 1)
            sigmas[i, 1] = np.log(np.exp(stdlonlon) - 1)

            # do inverse softsign on corlatlon because we later run softsign on corlatlon values in the neural network: softsign = x / (1 + abs(x))
            # later when softsign is applied on unprocessed_cor, corlatlon will be retrieved
            softsigncor = corlatlon / (1 + np.abs(corlatlon))
            raw_cor = corlatlon / (1.0 - corlatlon * np.sign(softsigncor))
            corxys[i] = raw_cor
        else:
            corxys[i] = corlatlon
            sigmas[i, 0] = stdlatlat
            sigmas[i, 1] = stdlonlon

    mus = kmns.cluster_centers_.astype('float32')

    return mus, sigmas, corxys 

def detect_nan(i, node, fn):
    for output in fn.outputs:
        if not isinstance(output[0], np.random.RandomState):
            if sp.sparse.issparse(output[0]):
                nans = np.isnan(output[0].data).any()
            else:
                nans = np.isnan(output[0]).any()
            if nans:
                print('*** NaN detected ***')
                theano.printing.debugprint(node)
                print('Inputs : %s' % [input[0] for input in fn.inputs])
                print('Outputs: %s' % [output[0] for output in fn.outputs])
                break

class NNModel_lang2locshared():
    def __init__(self,
                 n_epochs=10,
                 batch_size=1000,
                 regul_coef=1e-6,
                 input_size=None,
                 output_size=None,
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
                 mus=None,
                 sigmas=None,
                 corxy=None,
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
        self.mus = mus
        self.sigmas = sigmas
        self.corxy = corxy
        self.nan = False
        self.dataset_name = dataset_name
        logging.info('building nn model with %d hidden size, %d bivariate gaussian components and %d output size' % (self.hid_size, self.n_bigaus_comp, self.output_size))
        if self.sqerror:
            self.build_squarederror_regression()
        else:
            self.build()
            

    def nll_loss_sharedparams(self, mus, sigmas, corxy, pis, y_true):
        """
        negative log likelihood loss of a 2d y_true coordinate in
        each of the Gaussians with parameters mus, sigmas, corxy, pis.
        Note that the mus, sigmas and corxy are shared between all samples
        and only pis are different for each sample.
        
        The formula for negative log likelihood is :
        \mathcal{L}(y \vert x) = - \log\bigg\{\sum_{k=1}^K \pi_k(x)  \mathcal{N}\big(y \vert \mu_k(x), \Sigma_k(x)\big)\bigg\}
        
        The size of pis is n_batch x n_components,
        the size of mus is n_components x 2,
        the size of sigmas is n_components x 2 and
        the size of corxy is n_components x 1.
        
        The size of y_true is batch_size x 2.
        """
        mus_ex = mus[np.newaxis, :, :]
        X = y_true[:, np.newaxis, :]
        diff = X - mus_ex
        diffprod = T.prod(diff, axis=-1)
        corxy2 = corxy ** 2
        diff2 = diff ** 2
        sigmas2 = sigmas ** 2
        sigmainvs = 1.0 / sigmas
        sigmainvprods = sigmainvs[:, 0] * sigmainvs[:, 1]
        diffsigma = diff2 / sigmas2
        diffsigmanorm = T.sum(diffsigma, axis=-1)
        z = diffsigmanorm - 2 * corxy * diffprod * sigmainvprods
        oneminuscorxy2inv = 1.0 / (1.0 - corxy2)
        expterm = -0.5 * z * oneminuscorxy2inv
        #apply logsumExp trick for numerical stability https://hips.seas.harvard.edu/blog/2013/01/09/computing-log-sum-exp/
        new_exponent = T.log(0.5 / np.pi) + T.log(sigmainvprods) + T.log(np.sqrt(oneminuscorxy2inv)) + expterm + T.log(pis)
        max_exponent = T.max(new_exponent , axis=1, keepdims=True)
        mod_exponent = new_exponent - max_exponent
        gauss_mix = T.sum(T.exp(mod_exponent), axis=1)
        log_gauss = max_exponent + T.log(gauss_mix)
        loss = -T.mean(log_gauss)
        return loss

    def pred(self, mus, sigmas, corxy, pis, prediction_method='mixture'):
        '''
        select mus that maximize \sum_{pi_i * prob_i(mu)} if mean_prediction is True
        else
        select the component with the highest prob (ignore sum and pis)
        '''
        if prediction_method == 'mixture':
            # logging.info('predicting the best mixture mus')
            X = mus[:, :, :, np.newaxis]
            musex = mus[:, :, np.newaxis, :]
            sigmasex = sigmas[:, :, :, np.newaxis]
            corxysex = corxy[:, :, np.newaxis]
            diff = X - musex
            diffprod = np.prod(diff, axis=-3)
            sigmainvs = 1.0 / sigmasex
            sigmainvprods = sigmainvs[:, 0, :, :] * sigmainvs[:, 1, :, :]
            sigmas2 = sigmas ** 2
            corxy2 = corxysex ** 2
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
            selected_mus = mus[np.arange(mus.shape[0]), :, preds]
            
            # selected_sigmas = sigmas[np.arange(sigmas.shape[0]),preds,:]
            # selected_corxy = corxy[np.arange(corxy.shape[0]),preds]
            # selected_pis = pis[np.arange(pis.shape[0]),preds]        
            return selected_mus
        elif prediction_method == 'pi':
            # logging.info(sigmas[0])
            # logging.info(pis[0])
            # logging.info(corxy[0])
            
            logging.info('only pis are used for prediction')
            preds = np.argmax(pis, axis=1)
            selected_mus = mus[np.arange(mus.shape[0]), :, preds]
            # selected_sigmas = sigmas[np.arange(sigmas.shape[0]), :, preds]
            # selected_corxy = corxy[np.arange(corxy.shape[0]),preds]
            # selected_pis = pis[np.arange(pis.shape[0]),preds]        
            return selected_mus
        elif prediction_method == 'mixture':
            logging.info('not implemented!')
    
    def pred_sharedparams(self, mus, sigmas, corxy, pis, prediction_method='mixture'):
        """
        Given a mixture of Gaussians infer a mu that maximizes the mixture.
        There are two modes:
        If prediction_method==mixture then predict one of the mus that maximizes
        \mathcal{P}(\boldsymbol{x}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\boldsymbol{x} \vert \boldsymbol{\mu_k}, \Sigma_k)
        
        If prediction_method==pi return the mu that has the largest pi.
        """

        if prediction_method == 'mixture':
            X = mus[:, np.newaxis, :]
            diff = X - mus
            diffprod = np.prod(diff, axis=-1)
            sigmainvs = 1.0 / sigmas
            sigmainvprods = sigmainvs[:, 0] * sigmainvs[:, 1]
            sigmas2 = sigmas ** 2
            corxy2 = corxy ** 2
            diff2 = diff ** 2
            diffsigma = diff2 / sigmas2
            diffsigmanorm = np.sum(diffsigma, axis=-1)
            z = diffsigmanorm - 2 * corxy * diffprod * sigmainvprods
            oneminuscorxy2inv = 1.0 / (1.0 - corxy2)
            term = -0.5 * z * oneminuscorxy2inv
            expterm = np.exp(term)
            probs = (0.5 / np.pi) * sigmainvprods * np.sqrt(oneminuscorxy2inv) * expterm
            piprobs = pis[:, np.newaxis, :] * probs
            piprobsum = np.sum(piprobs, axis=-1)
            preds = np.argmax(piprobsum, axis=1)
            selected_mus = mus[preds, :]
            return selected_mus
        
        elif prediction_method == 'pi':
            logging.info('only pis are used for prediction')
            preds = np.argmax(pis, axis=1)
            selected_mus = mus[preds, :]
            #selected_sigmas = sigmas[np.arange(sigmas.shape[0]), :, preds]
            #selected_corxy = corxy[np.arange(corxy.shape[0]),preds]
            #selected_pis = pis[np.arange(pis.shape[0]),preds]        
            return selected_mus

  
            
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
            """
            sigmainvs = 1.0 / sigmas
            sigmainvprods = sigmainvs[:,:, 0] * sigmainvs[:,:, 1]
            sigmas2 = sigmas ** 2
            corxy2 = corxy **2
            diff2 = diff ** 2
            diffsigma = diff2 / sigmas2
            diffsigmanorm = np.sum(diffsigma, axis=-1)
            z = diffsigmanorm - 2 * corxy * diffprod * sigmainvprods
            oneminuscorxy2inv = 1.0 / (1.0 - corxy2)
            expterm = np.exp(-0.5 * z * oneminuscorxy2inv)
            expterm = 1.0
            probs = (0.5 / np.pi) * sigmainvprods * T.sqrt(oneminuscorxy2inv) * expterm
            probs = pis * probs
            """
            logging.fatal("not implemented!")
            sys.exit()
        elif prediction_method == "pi":    
            preds = T.argmax(pis, axis=1)
            selected_mus = mus[T.arange(mus.shape[0]), preds, :]
            return selected_mus


    
    def build(self):
        """
        build the MDN network with shared Gaussian parameters
        Input is sparse text and output is the parameters of the mixture of Gaussian
        """
        self.X_sym = S.csr_matrix(name='inputs', dtype=self.dtype)
        self.Y_sym = T.matrix(name='y_true', dtype=self.dtype)

        l_in_text = lasagne.layers.InputLayer(shape=(None, self.input_size),
                                         input_var=self.X_sym)

        if self.drop_out and self.dropout_coef > 0:
            l_in_text = lasagne_layers.SparseInputDropoutLayer(l_in_text, p=self.dropout_coef)


        l_hid_text = SparseInputDenseLayer(l_in_text, num_units=self.hid_size,
                                      nonlinearity=lasagne.nonlinearities.tanh,
                                      W=lasagne.init.GlorotUniform())

        # if self.drop_out and self.dropout_coef > 0:
        #    l_hid_text = lasagne.layers.dropout(l_hid_text, p=self.dropout_coef)

        self.l_pi_out = lasagne_layers.MDNSharedParams(l_hid_text, num_units=self.n_bigaus_comp,
                                                       mus=self.mus, sigmas=self.sigmas, corxy=self.corxy,
                                                       nonlinearity=lasagne.nonlinearities.softmax,
                                                       W=lasagne.init.GlorotUniform())
        
            
        pis = lasagne.layers.get_output(self.l_pi_out, self.X_sym)
        #use the shared gaussian parameters of the layer
        mus, sigmas, corxy = self.l_pi_out.mus, self.l_pi_out.sigmas, self.l_pi_out.corxy
        sigmas = T.nnet.softplus(sigmas)
        corxy = T.nnet.nnet.softsign(corxy)        
        loss = self.nll_loss_sharedparams(mus, sigmas, corxy, pis, self.Y_sym)
        #we can add an autoencoder loss if we want here
        #sq_error_coef = 0.01
        #predicted_mu = self.get_symb_mus(mus, sigmas, corxy, pis, prediction_method="pi")
        #loss += lasagne.objectives.squared_error(predicted_mu, self.Y_sym).mean() * sq_error_coef
            
        #if regul_coef is more than 0 apply regularization
        if self.regul_coef:
            l1_share_out = 0.5
            l1_share_hid = 0.5
            regul_coef_out, regul_coef_hid = self.regul_coef, self.regul_coef
            logging.info('regul coefficient for output and hidden lasagne_layers is ' + str(self.regul_coef))
            l1_penalty = lasagne.regularization.regularize_layer_params(self.l_pi_out, l1) * regul_coef_out * l1_share_out
            l2_penalty = lasagne.regularization.regularize_layer_params(self.l_pi_out, l2) * regul_coef_out * (1 - l1_share_out)
            l1_penalty += lasagne.regularization.regularize_layer_params(l_hid_text, l1) * regul_coef_hid * l1_share_hid
            l2_penalty += lasagne.regularization.regularize_layer_params(l_hid_text, l2) * regul_coef_hid * (1 - l1_share_hid)

            loss += l1_penalty + l2_penalty


        
        parameters = lasagne.layers.get_all_params(self.l_pi_out, trainable=True)
        updates = lasagne.updates.adam(loss, parameters, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8)
        self.f_train = theano.function([self.X_sym, self.Y_sym], loss, updates=updates, on_unused_input='warn')  # ,  mode=theano.compile.MonitorMode(pre_func=inspect_inputs, post_func=inspect_outputs))
        self.f_val = theano.function([self.X_sym, self.Y_sym], loss, on_unused_input='warn')
        self.f_predict = theano.function([self.X_sym], [mus, sigmas, corxy, pis], on_unused_input='warn')


    def build_squarederror_regression(self):
        """
        This is only used if we want to build a regression model
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

        #if self.drop_out and self.dropout_coef > 0:
        #    l_hid_text = lasagne.layers.dropout(l_hid_text, p=self.dropout_coef)

 
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
            l1_penalty = lasagne.regularization.regularize_layer_params(self.l_out, l1) * regul_coef_out * l1_share_out
            l2_penalty = lasagne.regularization.regularize_layer_params(self.l_out, l2) * regul_coef_out * (1 - l1_share_out)
            l1_penalty += lasagne.regularization.regularize_layer_params(l_hid_text, l1) * regul_coef_hid * l1_share_hid
            l2_penalty += lasagne.regularization.regularize_layer_params(l_hid_text, l2) * regul_coef_hid * (1 - l1_share_hid)


            loss = loss + l1_penalty + l2_penalty

        parameters = lasagne.layers.get_all_params(self.l_out, trainable=True)
        updates = lasagne.updates.adam(loss, parameters, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8)
        self.f_train = theano.function([self.X_sym, self.Y_sym], loss, updates=updates, on_unused_input='warn')
        self.f_val = theano.function([self.X_sym, self.Y_sym], loss, on_unused_input='warn')
        self.f_predict = theano.function([self.X_sym], output_eval, on_unused_input='warn')
        
    def set_params(self, params):
        lasagne.layers.set_all_param_values(self.l_pi_out, params)

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
        model_file = './data/lang2locshared_%s_hid%d_gaus%d.pkl' % (self.dataset_name, self.hid_size, self.n_bigaus_comp)
        if self.reload:
            if path.exists(model_file):
                logging.info('loading the model from %s' % model_file)
                with open(model_file, 'rb') as fin:
                    params = pickle.load(fin)
                self.set_params(params)
                return

        
        logging.info('training with %d n_epochs and  %d batch_size' % (self.n_epochs, self.batch_size))
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
                # latlon_pred = self.predict(x_batch)
                # logging.info(latlon_pred[0])
            l_vals = []
            for batch in self.iterate_minibatches(X_dev, Y_dev, self.batch_size, shuffle=False):
                x_batch, y_batch = batch
                l_val = self.f_val(x_batch, y_batch)
                l_vals.append(l_val)
            l_val = np.mean(l_vals)
            if np.isnan(l_val):
                self.nan = True
                return None

            if l_val < best_val_loss:
                best_val_loss = l_val
                if self.sqerror:
                    best_params = lasagne.layers.get_all_param_values(self.l_pi_out)
                else:
                    best_params = lasagne.layers.get_all_param_values(self.l_pi_out)
                n_validation_down = 0
            else:
                n_validation_down += 1
                if n_validation_down > self.early_stopping_max_down:
                    logging.info('validation results went down. early stopping ...')
                    break
            logging.info('iter %d, train loss %f, dev loss %f, best dev loss %f, num_down %d' % (step, l_train, l_val, best_val_loss, n_validation_down))
        if self.sqerror:
            lasagne.layers.set_all_param_values(self.l_out, best_params)
        else:
            lasagne.layers.set_all_param_values(self.l_pi_out, best_params)
        logging.info('dumping the model...')
        with open(model_file, 'wb') as fout:
            pickle.dump(best_params, fout)
                
    def predict(self, X):
        mus_eval, sigmas_eval, corxy_eval, pis_eval = self.f_predict(X)
        mus_eval, sigmas_eval, corxy_eval, pis_eval = np.asarray(mus_eval), np.asarray(sigmas_eval), np.asarray(corxy_eval), np.asarray(pis_eval)
        selected_mus = self.pred_sharedparams(mus_eval, sigmas_eval, corxy_eval, pis_eval)
        return selected_mus       

    def predict_regression(self, X):
        output = self.f_predict(X)
        return output

dl = None
         
def load_data(data_home, **kwargs):
    global dl
    bucket_size = kwargs.get('bucket', 500)
    dataset_name = kwargs.get('dataset_name')
    encoding = kwargs.get('encoding', 'utf-8')
    celebrity_threshold = kwargs.get('celebrity', 10)  
    mindf = kwargs.get('mindf', 10)
    dtype = kwargs.get('dtype', 'float32')
    one_hot_label = kwargs.get('onehot', False)

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
    cov = np.array([[1, 0], [0, 1]])
    # create bivariate gaussians to sample from the means (with variances 1, 1 and correlation 0) Melb, Au samples are two times of Melb, FL
    mlb_fl_samples = np.random.multivariate_normal(mean=mlb_fl_latlon_mean, cov=cov, size=n_samples).astype(dtype)
    mlb_au_samples = np.random.multivariate_normal(mean=mlb_au_latlon_mean, cov=cov, size=n_samples * 2).astype(dtype)
    
    # plt.scatter(mlb_fl_samples[:, 0], mlb_fl_samples[:, 1], c='blue', s=1)
    # plt.scatter(mlb_au_samples[:, 0], mlb_au_samples[:, 1], c='red', s=1)
    # plt.show()

    X = sp.sparse.csr_matrix(np.random.uniform(-0.1, 0.1, size=(n_samples * 3, 2)) + np.array([1, 0])).astype(dtype)
    
    Y = np.vstack((mlb_fl_samples, mlb_au_samples))  
    # shuffle X and Y
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
    data = (X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, None, None, userLocation, None)
    return data
    
    
    
def train(data, **kwargs):
    dropout_coef = kwargs.get('dropout_coef', 0.5)
    regul = kwargs.get('regul_coef', 1e-6)
    hid_size = kwargs.get('hidden_size', 200)
    autoencoder = kwargs.get('autoencoder', False)
    grid_transform = kwargs.get('grid', False)
    rbf = kwargs.get('rbf', False)
    ncomp = kwargs.get('ncomp', 100)
    dataset_name = kwargs.get('dataset_name')
    sqerror = kwargs.get('sqerror', False)
    batch_size = kwargs.get('batch_size', 200 if dataset_name == 'cmu' else 2000)
    # X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation, loc_train = data
    input_size = data[0].shape[1]
    output_size = data[1].shape[1] if len(data[1].shape) == 2 else np.max(data[1]) + 1
    logging.info('batch size %d' % batch_size)
    max_down = 20 if dataset_name == 'cmu' else 5 
    mus, raw_stds, raw_cors = get_cluster_centers(data[12], n_cluster=ncomp)
    # just set the mus let sigmas and corxys to be initialised!
    raw_stds, raw_cors = None, None
    model = NNModel_lang2locshared(n_epochs=10000, batch_size=batch_size, regul_coef=regul,
                    input_size=input_size, output_size=output_size, hid_size=hid_size,
                    drop_out=True, dropout_coef=dropout_coef, early_stopping_max_down=max_down,
                    input_sparse=True, reload=False, ncomp=ncomp, autoencoder=autoencoder, sqerror=sqerror,
                    mus=mus, sigmas=raw_stds, corxy=raw_cors, dataset_name=dataset_name)

    model.fit(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[11])
    if model.nan:
        logging.info('nan occurred')
        return 0, 0, 0
    # save some space before prediction

    if model.sqerror:
        latlon_pred = model.predict_regression(data[2])
    else:
        if dataset_name == 'cmu':
            latlon_pred = model.predict(data[2])
        else:
            latlon_preds = []
            for batch in model.iterate_minibatches(data[2], data[2], model.batch_size, shuffle=False):
                x_batch, x_batch = batch
                latlon_pred = model.predict(x_batch)
                latlon_preds.append(latlon_pred)
            latlon_pred = np.vstack(tuple(latlon_preds))
    logging.info('dev results:')
    mean , median, acc = geo_latlon_eval(data[7], data[11], latlon_pred, contour_error_on_map=False)
    
    if model.sqerror:
        latlon_pred = model.predict_regression(data[4])
    else:
        if dataset_name == 'cmu':
            latlon_pred = model.predict(data[4])
        else:
            #you can't predict for all dev samples because of memory size
            latlon_preds = []
            for batch in model.iterate_minibatches(data[4], data[4], model.batch_size, shuffle=False):
                x_batch, x_batch = batch
                latlon_pred = model.predict(x_batch)
                latlon_preds.append(latlon_pred)
            latlon_pred = np.vstack(tuple(latlon_preds))
    logging.info('test results:')
    mean_test , median_test, acc_test = geo_latlon_eval(data[8], data[11], latlon_pred)
    return mean, median, acc, mean_test, median_test, acc_test


def tune(data, dataset_name, args, num_iter=100):
    logging.info('tuning over %s' % dataset_name)
    param_scores = []
    random.seed()
    for ncomp in [100, 300, 900]:
        for hidden_size in [100, 300, 900]:
            if hidden_size > ncomp:
                continue
            for regul_coef in [0, 1e-5]:
                for drop_out_ceof in [0, 0.5]:
                    np.random.seed(77)    
                    logging.info('regul %f drop %f hidden %d ncomp %d' % (regul_coef, drop_out_ceof, hidden_size, ncomp))
                    try:
                        mean, median, acc, mean_test, median_test, acc_test = train(data, regul_coef=regul_coef, dropout_coef=drop_out_ceof, hidden_size=hidden_size, ncomp=ncomp, dataset_name=dataset_name, sqerror=args.sqerror)
                    except:
                        logging.info('exception occurred')
                        continue
            
                    scores = OrderedDict()
                    scores['mean_dev'], scores['median_dev'], scores['acc_dev'] = mean, median, acc
                    scores['mean_test'], scores['median_test'], scores['acc_test'] = mean_test, median_test, acc_test
                    params = OrderedDict()
                    params['regul'], params['dropout'], params['hidden'], params['ncomp'] = regul_coef, drop_out_ceof, hidden_size, ncomp
                    param_scores.append([params, scores])
                    logging.info(params)
                    logging.info(scores)
    for param_score in param_scores:
        logging.info(param_score)

def fine_tune(data, dataset_name, args, num_iter=100):
    logging.info('tuning over %s' % dataset_name)
    param_scores = []
    random.seed()
    if dataset_name == 'cmu':
        ncomp = 300
        hidden_size = 100
    else:
        ncomp = 900
        hidden_size = 900
    for regul_coef in [0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
        for drop_out_ceof in [0, 0.3, 0.4, 0.5, 0.6, 0.7]:
            np.random.seed(77)    
            logging.info('regul %f drop %f hidden %d ncomp %d' % (regul_coef, drop_out_ceof, hidden_size, ncomp))
            try:
                mean, median, acc, mean_test, median_test, acc_test = train(data, regul_coef=regul_coef, dropout_coef=drop_out_ceof, hidden_size=hidden_size, ncomp=ncomp, dataset_name=dataset_name, sqerror=args.sqerror)
            except:
                logging.info('exception occurred')
                continue
    
            scores = OrderedDict()
            scores['mean_dev'], scores['median_dev'], scores['acc_dev'] = mean, median, acc
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
    parser.add_argument('-i', '--dataset', metavar='str', help='dataset for dialectology', type=str, default='na')
    parser.add_argument('-bucket', '--bucket', metavar='int', help='discretisation bucket size', type=int, default=300)
    parser.add_argument('-batch', '--batch', metavar='int', help='SGD batch size', type=int, default=0)
    parser.add_argument('-hid', '--hidden', metavar='int', help='Hidden layer size', type=int, default=500)
    parser.add_argument('-mindf', '--mindf', metavar='int', help='minimum document frequency in BoW', type=int, default=10)
    parser.add_argument('-d', '--dir', metavar='str', help='home directory', type=str, default='./data')
    parser.add_argument('-enc', '--encoding', metavar='str', help='Data Encoding (e.g. latin1, utf-8)', type=str, default='utf-8')
    parser.add_argument('-reg', '--regularization', metavar='float', help='regularization coefficient)', type=float, default=1e-6)
    parser.add_argument('-drop', '--dropout', metavar='float', help='dropout coef default 0.5', type=float, default=0.5)
    parser.add_argument('-cel', '--celebrity', metavar='int', help='celebrity threshold', type=int, default=10)
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
    # THEANO_FLAGS='device=cpu' nice -n 10 python lang2loc_mdnshared.py -d ~/datasets/na/processed_data/ -enc utf-8 -reg 0.0 -drop 0.0 -mindf 10 -hid 900 -ncomp 900 -batch 2000
    # THEANO_FLAGS='device=cpu' nice -n 10 python lang2loc_mdnshared.py -d ~/datasets/cmu/processed_data/ -enc latin1 -reg 0.0 -drop 0.0 -mindf 10 -hid 100 -ncomp 300 -batch 200 
    args = parse_args(sys.argv[1:])
    datadir = args.dir
    dataset_name = datadir.split('/')[-3]
    logging.info('dataset: %s' % dataset_name)
    
    if args.toy:
        logging.info('toy dataset is being used.')
        data = load_toy_data()
    else:
        data = load_data(data_home=args.dir, encoding=args.encoding, mindf=args.mindf, grid=args.grid, dataset_name=dataset_name)
    
    if args.batch:
        batch_size = args.batch
    else:
        batch_size = 200 if dataset_name == 'cmu' else 2000
    
    if args.tune:
        tune(data, dataset_name, args)
    else:
        train(data, regul_coef=args.regularization, dropout_coef=args.dropout,
              hidden_size=args.hidden, autoencoder=args.autoencoder, grid=args.grid, rbf=args.rbf,
              ncomp=args.ncomp, dataset_name=dataset_name, sqerror=args.sqerror, batch_size=batch_size)
