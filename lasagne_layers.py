'''
Created on 3 Feb 2017

@author: af
'''
import numpy as np
import theano.tensor as T
import theano.sparse as S
from lasagne.layers import DenseLayer, DropoutLayer, Layer
from collections import OrderedDict

'''
These sparse classes are copied from https://github.com/Lasagne/Lasagne/pull/596/commits
'''
class SparseInputDenseLayer(DenseLayer):
    """A dense layer suitable to be placed after sparse input layers
    copied from https://github.com/Lasagne/Lasagne/pull/596/commits
    """
    def get_output_for(self, input, **kwargs):
        if not isinstance(input, (S.SparseVariable, S.SparseConstant,
                                  S.sharedvar.SparseTensorSharedVariable)):
            raise ValueError("Input for this layer must be sparse")

        activation = S.dot(input, self.W)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)

class SparseInputDropoutLayer(DropoutLayer):
    """A dropout layer suitable to be placed after an input layer with sparse data
    copied from https://github.com/Lasagne/Lasagne/pull/596/commits
    """
    def get_output_for(self, input, deterministic=False, **kwargs):
        if not isinstance(input, (S.SparseVariable, S.SparseConstant,
                                  S.sharedvar.SparseTensorSharedVariable)):
            raise ValueError("Input for this layer must be sparse")

        if deterministic or self.p == 0:
            return input
        else:
            # Using Theano constant to prevent upcasting
            one = T.constant(1, name='one')
            retain_prob = one - self.p

            if self.rescale:
                input = S.mul(input, one/retain_prob)

            input_shape = self.input_shape
            if any(s is None for s in input_shape):
                input_shape = input.shape

            return input * self._srng.binomial(input_shape, p=retain_prob,
                                               dtype=input.dtype)


class GaussianRBFLayer(Layer):
    """
    Given 2d inputs this layer can return the RBF activation of each input
    in each of the num_units 2d RBFs. It can easily be extended to inputs
    with higher/lower dimensionality by changing the mus_init dimensionality.
    """
    def __init__(self, incoming, num_units, mus=None, **kwargs):
        super(GaussianRBFLayer, self).__init__(incoming, **kwargs)
        #self.H = H
        #self.H = self.add_param(H, (H.shape[0], H.shape[1]), name='H')
        self.name = 'GaussianRBFLayer'
        self.num_units = num_units
        if mus is not None:
            self.mus_init = mus
        else:
            self.mus_init = np.random.randn(self.num_units, 2).astype('float32')
        
        self.sigmas_init = np.abs(np.random.randn(self.num_units).reshape((self.num_units,))).astype('float32')
        
        self.mus = self.add_param(self.mus_init, self.mus_init.shape, name='mus')
        self.sigmas = self.add_param(self.sigmas_init, self.sigmas_init.shape, name='sigmas')

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)        
    
    def get_output_for(self, input, **kwargs):
        """
        Given n-dimensional input and the current mus, find the RBF activation of
        each input in each RBF component using RBF(x,mu,sigma) =e^{- \sigma_i(x_i-mu_i)^2 * sigma^2}
        """
        C = self.mus[np.newaxis, :, :]
        X = input[:, np.newaxis, :]
        difnorm = T.sum((C-X)**2, axis=-1)
        a = T.exp(-difnorm * (self.sigmas**2))
        return a

class DiagonalBivariateGaussianLayer(Layer):
    """
    Diagonal Bivariate Gaussian Activation Layer
    Given 2d input the output of this layer is the probability
    of each input in each of the num_units Diagonal Bivariate Gaussians
    using this formula http://mathworld.wolfram.com/BivariateNormalDistribution.html
    It is fairly easy to customize the class for higher/lower dimensionality
    """
    def __init__(self, incoming, num_units, mus=None, **kwargs):
        super(DiagonalBivariateGaussianLayer, self).__init__(incoming, **kwargs)
        self.name = 'DiagonalBivariateGaussianLayer'
        self.num_units = num_units
        if mus is not None:
            self.mus_init = mus
        else:
            self.mus_init = np.random.randn(self.num_units, 2).astype('float32')
        
        self.sigmas_init = np.abs(np.random.randn(self.num_units, 2).reshape((self.num_units,2))).astype('float32')
        
        self.mus = self.add_param(self.mus_init, self.mus_init.shape, name='mus', regularizable=False)
        self.sigmas = self.add_param(self.sigmas_init, self.sigmas_init.shape, name='sigmas', regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)        
    
    def get_output_for(self, input, **kwargs):
        """
        Given 2d input find the probability of each input in each of num_units
        Diagonal Gaussians using the formula from http://mathworld.wolfram.com/BivariateNormalDistribution.html
        """
        #make sure sigma is positive and nonzero softplus(x) (0, +inf)
        sigmas = T.nnet.softplus(self.sigmas)
        sigmainvs = 1.0 / sigmas
        sigmainvprods = sigmainvs[:, 0] * sigmainvs[:, 1]
        sigmas2 = sigmas ** 2
        mus = self.mus[np.newaxis, :, :]
        X = input[:, np.newaxis, :]
        diff = (X - mus) ** 2
        diffsigma = diff / sigmas2
        diffsigmanorm = T.sum(diffsigma, axis=-1)
        expterm = T.exp(-0.5 * diffsigmanorm)
        probs = (0.5 / np.pi) * sigmainvprods * expterm
        return probs
        

class BivariateGaussianLayer(Layer):
    """
    Bivariate Gaussian Activation Layer
    Given 2d input the output of this layer is the probability
    of each input in each of the num_units  Bivariate Gaussians
    using this formula http://mathworld.wolfram.com/BivariateNormalDistribution.html
    It is fairly easy to customize the class for higher/lower dimensionality
    """
    def __init__(self, incoming, num_units, mus=None, sigmas=None, corxy=None, **kwargs):
        super(BivariateGaussianLayer, self).__init__(incoming, **kwargs)
        self.name = 'BivariateGaussianLayer'
        self.num_units = num_units
        if mus is not None:
            mus_init = mus
        else:
            mus_init = np.random.randn(self.num_units, 2).astype('float32')
        
        if sigmas is not None:
            sigmas_init = sigmas
        else:
            #sigmas_init = np.array([10, 10]).astype('float32') * np.abs(np.random.randn(self.num_units, 2).reshape((self.num_units,2))).astype('float32')
            sigmas_init = np.array([5, 5]).astype('float32') * np.ones(shape=(self.num_units, 2)).astype('float32')
        
        if corxy is not None:
            corxy_init = corxy
        else:
            corxy_init = np.random.randn(self.num_units,).reshape((self.num_units,)).astype('float32')
        
        self.mus = self.add_param(mus_init, mus_init.shape, name='mus', regularizable=False)
        self.sigmas = self.add_param(sigmas_init, sigmas_init.shape, name='sigmas', regularizable=False)
        self.corxy = self.add_param(corxy_init, corxy_init.shape, name='corxy', regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)        
    
    def get_output_for(self, input, do_softmax=True, **kwargs):
        """
        Return the probability of each input sample in
        each of the num_units Bivariate Gaussian components.
        Note that if do_softmax=True a softmas would be applied
        over those probabilities so that it can be used as a stable
        representation instead of many very small numbers. 
        If do_softmax=False the returned probabilites
        might be very small due to high distance from the means.
        """
        #make sure sigma is positive and nonzero softplus(x) (0, +inf)
        sigmas = T.nnet.softplus(self.sigmas)
        sigmainvs = 1.0 / sigmas
        sigmainvprods = sigmainvs[:, 0] * sigmainvs[:, 1]
        sigmas2 = sigmas ** 2
        #convert to 3d so that we can broadcast input n_batchxdimension, hidden n_hidxdimension to n_batchx1
        mus = self.mus[np.newaxis, :, :]
        X = input[:, np.newaxis, :]
        #difference between input and means for each dimension
        diff = X-mus
        #multiply x-mu1 , x-mu2 the result should be number_of_samples x number_of_hidden
        diffprod = T.prod(diff, axis=-1)
        #correlation x, y member of (-1, 1)  softsign(x) = x / (1 + abs(x))
        corxy = T.nnet.nnet.softsign(self.corxy)
        #power 2 of corxy
        corxy2 = corxy **2
        diff2 = diff ** 2
        diffsigma = diff2 / sigmas2
        diffsigmanorm = T.sum(diffsigma, axis=-1)
        z = diffsigmanorm - 2 * corxy * diffprod * sigmainvprods
        oneminuscorxy2inv = 1.0 / (1.0 - corxy2)
        expterm = -0.5 * z * oneminuscorxy2inv
        #a trick for numerical stability reasons: a1 * a2 * exp(b) == exp(log(a1) + log(a2) + b)
        #probs = (0.5 / np.pi) * sigmainvprods * T.sqrt(oneminuscorxy2inv) * T.exp(expterm)    #the term without the logexp trick
        #note a x exp(b) = exp(log(a) + b)
        logprobs = T.log(0.5/np.pi) + T.log(sigmainvprods) + T.log(T.sqrt(oneminuscorxy2inv)) + expterm
        probs = T.exp(logprobs)
        if do_softmax:
            output = T.nnet.softmax(logprobs)
        else:
            output = probs
        return output

class MDNSharedParams(DenseLayer):
    """
    This is an implementation of Mixture Density Networks Bishop (2005) with
    the difference that mus and sigmas are shared between all input samples
    by setting them as the parameters of the layer. Only pis are conditioned
    on input.
    """
    def __init__(self, incoming, mus=None, sigmas=None, corxy=None, **kwargs):
        super(MDNSharedParams, self).__init__(incoming, **kwargs)
        #self.H = H
        #self.H = self.add_param(H, (H.shape[0], H.shape[1]), name='H')
        self.name = 'MDNSharedParams'

        if mus is not None:
            mus_init = mus
        else:
            mus_init = np.random.randn(self.num_units, 2).astype('float32')
        
        if sigmas is not None:
            sigmas_init = sigmas
        else:
            #emnlp submission
            sigmas_init = np.array([10, 10]).astype('float32') * np.abs(np.random.randn(self.num_units, 2).reshape((self.num_units,2))).astype('float32')
            #Note that sigma initialization is a very difficult task.
            #if they're very small, the probabilites in all gaussians will be close
            #to zero which results in numerical instability and nans/infs.
            #For each indivisual application the initialization should be
            #thoroughly customized.
            #e.g. sigmas_init = np.random.uniform(low=5, high=10, size=(self.num_units, 2)).astype('float32')
        
        if corxy is not None:
            corxy_init = corxy
        else:
            corxy_init = np.random.randn(self.num_units,).reshape((self.num_units,)).astype('float32')
        
        self.mus = self.add_param(mus_init, mus_init.shape, name='mus', regularizable=False)
        self.sigmas = self.add_param(sigmas_init, sigmas_init.shape, name='sigmas', regularizable=False)
        self.corxy = self.add_param(corxy_init, corxy_init.shape, name='corxy', regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)        
    
    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        activation = T.dot(input, self.W)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)
    
