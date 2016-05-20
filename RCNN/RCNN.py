import numpy as np
import lasagne as nn
from lasagne.layers import BatchNormLayer
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import Pool2DLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import ElemwiseSumLayer as SumLayer
from lasagne.nonlinearities import very_leaky_rectify as vRELU   
from lasagne.layers import NonlinearityLayer 
from lasagne import updates
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator


def makeModel(nbChannels, length, nbClasses, nbRCL=5,
		 nbFilters=128, filtersize = 3, epochs=50, update='adam', update_learning_rate=None,
		 objective_l2=0.0025, earlystopping=False, patience=10, batch_size=64,
		 verbose=0,):
	"""
	Function to build RCNN

	X should be [trials x electrodes x 1 x samples]

	Parameters
	-----------
	nbChannels : int 
		Number of channels in the input signal 
	length : int 
		Number of samples in the input time series (e.g., 1 second of 512 Hz is 512 samples)
	nbClasses : int
		Number of potential classes for the data
	nbRCL : int, optional
		Number of recurrent convolutional layers, defaults to 5
	nbFilters : int, optional
		Number of filters to use in convolutional layers, defaults to 128
	filtersize : int, opitional
		Width of the filter, should be an odd number. Defaults to 3.
	epochs : int, optional
		Number of Epochs to use for training, defaults to 50
	update : string, opitional ****FIX ME
		Which update to use (from lasagne), defaults to adam
	update_learning_rate : float, optional
		Learning rate, defaults to update default (see lasagne)
	objective_l2 : float, optional
		L2 regularization amount, default is 0.0025
	EarlyStopping : bool, optional
		Use Early stopping, defaults to False
	patience : int, optional
		If using early stopping, how many trials to wait before stopping, defaults to 10
	batch_size : int, optional
		batch size, default is 64
	verbose : int, optional
		1 to print info during training, 0 to silence output, default is 0

	"""
	l_out = BuildRCNN(nbChannels, length, nbClasses, nbRCL, nbFilters, filtersize)
	net = CompileNetwork(l_out, epochs, update, update_learning_rate, objective_l2,
				   earlystopping, patience, batch_size, verbose)

	return net


def BuildRCNN(nbChannels, length, nbClasses, nbRCL, nbFilters, filtersize):
    
    def RCL_block(l, pool=True, increase_dim=False):
	input_num_filters = l.output_shape[1]
	if increase_dim:
	   out_num_filters = input_num_filters*2
	else:
	   out_num_filters = input_num_filters

	stack1 = Conv2DLayer(incoming = l, num_filters = out_num_filters, filter_size = (1, 1), stride = 1, pad = 'same', W = nn.init.HeNormal(gain='relu'), nonlinearity = None)
	stack2 = BatchNormLayer(incoming = stack1)
	stack3 = NonlinearityLayer(stack2, nonlinearity=vRELU)
	stack4 = Conv2DLayer(incoming = stack3, num_filters = out_num_filters, filter_size = (1, filtersize), stride = 1, pad = 'same', W = nn.init.HeNormal(gain='relu'), b = None, nonlinearity = None)
	stack5 = SumLayer(incomings = [stack1, stack4], coeffs = 1)
	stack6 = BatchNormLayer(incoming = stack5 )
	stack7 = Conv2DLayer(incoming = stack6, num_filters = out_num_filters, filter_size = (1, filtersize), stride = 1, pad = 'same', W = stack4.W, b = None, nonlinearity = None)
	stack8 = SumLayer(incomings = [stack1, stack7], coeffs = 1)
	stack9 = BatchNormLayer(incoming = stack8 )
	stack10 = NonlinearityLayer(stack9, nonlinearity=vRELU)
	stack11 = Conv2DLayer(incoming = stack10, num_filters = out_num_filters, filter_size = (1, filtersize), stride = 1, pad = 'same', W = stack4.W, b = None, nonlinearity = None)
	stack12 = SumLayer(incomings = [stack1, stack11], coeffs = 1)
	stack13 = BatchNormLayer(incoming = stack12 )
	stack14 = NonlinearityLayer(stack13, nonlinearity=vRELU)
	if pool:
		stack15 = Pool2DLayer(incoming = stack14, pool_size = (1, 2))
		stack16 = nn.layers.DropoutLayer(incoming = stack15, p = 0.1)
	else: 
		stack16 = nn.layers.DropoutLayer(incoming = stack14, p = 0.1)

	return stack16

	#Build the network	
    l_in = InputLayer((None, nbChannels, 1, length))

	#Start with one normal convolutional layer
    l = Conv2DLayer(l_in , num_filters = nbFilters, filter_size = (1, filtersize), stride = 1, pad = 'same', W = nn.init.HeNormal(gain='relu'), nonlinearity = None)

	#Add n RCL stacks - Pooling happens after stacking two RCLs together 
	# (but divisible by 1 because we start with a normal conv layer)
    for n in range(nbRCL):
    	if n % 1 == 0:
			l = RCL_block(l, pool=False)
        else:
			l = RCL_block(l)



	#Fully connected layer
    l_out = DenseLayer(incoming = l, num_units = nbClasses,
                                 W = nn.init.HeNormal(gain='relu'),
                                 nonlinearity = nn.nonlinearities.softmax)

    return l_out

def CompileNetwork(l_out, epochs, update, update_learning_rate, objective_l2,
				   earlystopping, patience, batch_size, verbose):
	
    update_fn = getattr(updates, update)
    earlystop = EarlyStopping(patience=patience, verbose=verbose)

    net = NeuralNet(
        l_out,
        max_epochs=epochs,
         
        update=update_fn,
        
        objective_l2=objective_l2,
        
        batch_iterator_train = BatchIterator(batch_size=batch_size),
        batch_iterator_test = BatchIterator(batch_size=batch_size),    
        verbose=verbose,
        on_training_finished = [earlystop.load_best_weights]
    )
    
    if earlystopping == True: 
        net.on_epoch_finished.append(earlystop)
    if update_learning_rate is not None: 
        net.update_learning_rate=update_learning_rate
    

    return net


class EarlyStopping(object):
    def __init__(self, patience=10, verbose=0):
        self.patience = patience
        self.verbose= verbose
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping."),
            if self.verbose:
                print("Best valid loss was {:.6f} at epoch {}.".format(
                    self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()
    
    def load_best_weights(self, nn, train_history):
        print("Training completed.")
        if self.verbose:
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch)) 
        
        nn.load_params_from(self.best_weights)

				