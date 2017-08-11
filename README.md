
``GEOMDN`` readme
=================


Introduction
------------

``GEOMDN`` is an implementation of [Continuous Representation of Location for Geolocation and Lexical Dialectology using Mixture Density Networks (EMNLP2017)](https://drive.google.com/open?id=0B9ZfPKPvp-JickpYa0drZWQxcHc).

The neural-network is implemented using Theano/Lasagne but it shouldn't be difficult to adopt it to other
NN frameworks.

The work has 3 main sections:

1. lang2loc.py implements mixture density networks to predict location from text input

2. lang2loc_mdnshared.py implements mixture density networks to predict location from text input
with the difference that the mus, sigmas and corxys of the mixure of Gaussians are shared between
all the input samples and only pis of samples are conditioned on input. This improved the model as
the global mixture of Gaussian sturcture exists and can be learned from all the samples rather than
predicted for each individual sample.

3. loc2lang.py implements a lexical dialectology model where given 2d coordinate inputs predicts
a unigram probability distribution over vocabulary. The input is a normal 2d input layer but the
hidden layer consisits of several Gaussian distributions whose mus, sigmas and corxys are learned
and its output is the probability of input in each of the Gaussian components.

**Look at** some of the [maps](https://drive.google.com/open?id=0B9ZfPKPvp-JiWlhoZ01HMk9GY3c), a lot of
[local words](https://drive.google.com/open?id=0B9ZfPKPvp-JiTW1yWlF2ZG56SUE) including named entities for several DARE dialect regions 
and [city terms](https://drive.google.com/open?id=0B9ZfPKPvp-JiNHd6Um5nV2RBWjQ) including named entities for about 100 U.S. cities



Geolocation Datasets
--------------------
Datasets are GEOTEXT a.k.a CMU (a small Twitter geolocation dataset)
and TwitterUS a.k.a NA (a bigger Twitter geolocation dataset) both
covering continental U.S. which can be download from [here](https://www.amazon.com/clouddrive/share/kfl0TTPDkXuFqTZ17WJSnhXT0q6fGkTlOTOLZ9VVPNu)

Quick Start
-----------

1. Download the datasets and place them in ''./datasets/cmu'' and ''./datasets/na''
for GEOTEXT and TwitterUS (contact me for the datasets).

2. For lang2loc geolocation run:

For GEOTEXT a.k.a CMU run:

```sh
THEANO_FLAGS='device=cpu' nice -n 10 python lang2loc.py -d ./datasets/cmu/ -enc latin1 -reg 0 -drop 0.5 -mindf 10 -hid 100 -ncomp 100
```

For TwitterUS a.k.a NA run:

```sh
THEANO_FLAGS='device=cpu' nice -n 10 python lang2loc.py -d ./datasets/na/ -enc utf-8 -reg 1e-5 -drop 0.0 -mindf 10 -hid 300 -ncomp 100
```

3. For lang2loc_mdnshared geolocation run:

For GEOTEXT a.k.a CMU run:

```sh
THEANO_FLAGS='device=cpu' nice -n 10 python lang2loc_mdnshared.py -d ~/datasets/cmu/ -enc latin1 -reg 0.0 -drop 0.0 -mindf 10 -hid 100 -ncomp 300 -batch 200
```

For TwitterUS a.k.a NA run:

```sh
THEANO_FLAGS='device=cpu' nice -n 10 python lang2loc_mdnshared.py -d ~/datasets/na/ -enc utf-8 -reg 0.0 -drop 0.0 -mindf 10 -hid 900 -ncomp 900 -batch 2000
```


4. For loc2lang lexical dialectology model run:


```sh
THEANO_FLAGS='device=cpu'   nice -n 10 python loc2lang.py -d ~/datasets/na/ -enc utf-8 -reg 0.0 -drop 0.0 -mindf 100 -hid 1000 -ncomp 500 -batch 5000
```

Note that cmu is very small to be used for lexical dialectology.


Citation
--------
```
@InProceedings{rahimicontinuous2017,
  author    = {Rahimi, Afshin  and  Baldwin, Timothy and Cohn, Trevor},
  title     = {Continuous Representation of Location for Geolocation and Lexical Dialectology using Mixture Density Networks },
  booktitle = {Proceedings of Conference on Empirical Methods in Natural Language Processing (EMNLP2017)},
  month     = {September},
  year      = {2017},
  address   = {Copenhagen, Denmark},
  publisher = {Association for Computational Linguistics},
  url       = {http://people.eng.unimelb.edu.au/tcohn/papers/emnlp17geomdn.pdf}
}
```

Contact
-------
Afshin Rahimi <afshinrahimi@gmail.com>
