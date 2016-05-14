# RCNN Toolbox

by Jacob Zweig @ [Northwestern University Visual perception, Neuroscience, and Cognition Lab](http://groups.psych.northwestern.edu/grabowecky_suzuki/Grabowecky_Suzuki_Lab/Research.html)

For decoding electrophysiologic signals using recurrent convolutional neural networks. Uses ensemble voting for classification. Mostly written around [lasagne](http://lasagne.readthedocs.io/en/latest/index.html) and [scikit-learn](http://scikit-learn.org/). RCNNs are based on the architecture outlined in [Recurrent Convolutional Neural Network for Object Recognition](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liang_Recurrent_Convolutional_Neural_2015_CVPR_paper.pdf).

## Installation

It's easiest to start with a scientific distribution. If you're in academia [Enthought Canopy](https://enthought.com/products/canopy/) is free, and works well for me. People also like Anaconda.

### Dependencies:

First, `cd` to the ECoG_Decoding folder and run the install command:
```
pip install -r requirements.txt
```
If that doesn't work, install these manually:
- numpy, scipy
- pyyaml
- Theano: [installation instructions](http://deeplearning.net/software/theano/install.html#install) (Take special note of GPU installation instructions, because it will be really slow without it)
- Lasagne: ([Install instructions](http://lasagne.readthedocs.io/en/latest/user/installation.html))
- Optional but recommended if you use CNNs: cuDNN (will save you lots and lots of time!)


**Note**: You should use the latest version of Theano, not the PyPI version. Install it with:
```
sudo pip install git+git://github.com/Theano/Theano.git
```

I like to use Matlab to process my EEG data. If that's the case, you might consider using the Matlab Engine for Python since it makes transferring data between the two programs quite seamless. To install the Matlab Engine for Python, follow the [instructions here](http://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)


Finally, to install this package, `cd` to the RCNN-Toolbox folder and run the install command:
```
sudo python setup.py install
```

If you're using cuDNN, enable CNMeM in your .theanorc file (I set it to .75 and it seems to work well). 



## Documentation

In progress...





