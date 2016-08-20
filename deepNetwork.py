
from __future__ import print_function

import pickle
import sys
import os
import time

import numpy
import numpy as np
import pandas as pd

import theano
import theano.tensor as T
import lasagne



# prepare data for training
def load_dataset(dataFrame):
    X = []
    Y = []
    for idx in dataFrame:
        i=0
        if(i<3):
            X.append(dataFrame[idx])
            i+=1
        else:
            Y.append(dataFrame[idx])
    X = numpy.asarray(X, dtype=np.float32)
    Y = numpy.asarray(Y, dtype=np.float32)
    return X, Y

def iterate_minibatches(inputs, targets, 
                        inputDimns=3, outputDimns=12, 
                        batch_size=128, shuffle=False):
    assert len(inputs) == len(targets)
    assert inputs.shape[1] == inputDimns
    assert targets.shape[1] == outputDimns
    if shuffle:
        idxs = np.arange(inputs.shape[0])
        np.random.shuffle(idxs)
    for init_idx in range(0, input.shape[0]-batch_size+1, batch_size):
        if shuffle:
            data_slice = idxs[init_idx: init_idx+batch_size]
        else:
            data_slice = slice(init_idx, init_idx+batch_size)
        yield inputs[data_slice], targets[data_slice]


def build_MLP(input_var=None, featureDimn=3, outputDimns=12):
    input_layer = lasagne.layers.InputLayer(shape=(None, featureDimn), 
                                            input_var=input_var)
    hidden_layer = lasagne.layers.DenseLayer(input_layer, num_units=gaussNum*3, 
                                             nonlinearity=lasagne.nonlinearities.rectify, 
                                             W=lasagne.init.GlorotUniform())
    output_layer = lasagne.layers.DenseLayer(input_layer, num_units=gaussNum*3, 
                                             nonlinearity=lasagne.nonlinearities.rectify, 
                                             W=lasagne.init.GlorotUniform())
def load_dataFrames(trainFile, valFile, testFile):
    print('Loading data ...')
    train_data, val_data, test_data = (None, None, None)
    with open('converted_train.pkl', 'rb') as f:
        print('Loading training set ...')
        train_data = pickle.load(f)
        print(len(train_data))
        train_data = train_data.reset_index()
        print("Size of training set: "+str(len(train_data)))
    with open('converted_dev.pkl', 'rb') as f:
        print('Loading validation set ...')
        val_data = pickle.load(f)
        val_data = val_data.reset_index()
        print(len(val_data))
    with open('converted_test.pkl', 'rb') as f:
        print('Loading test set ...')
        test_data = pickle.load(f)
        test_data = test_data.reset_index()
        print(len(test_data))
    return train_data, val_data, test_data


def main(inputDimns=3, outputDimns=12, num_epochs=10, 
         batch_size=128, num_units=12, architecture='MLP', 
         lr=0.01, momentum=0.9, drop_out=0., optimizer='SGD', 
         momentumType='Nesterov'):
    #load data frames from pickled files
    train_data, val_data, test_data = load_dataFrame('converted_train.pkl', 
                                                     'converted_dev.pkl', 
                                                     'converted_test.pkl')
    #load data sets into GPU-capable data types
    X_train, Y_train = load_dataset(train_data)
    X_val, Y_val = load_dataset(val_data)
    X_test, Y_test = load_dataset(test_data)
    #initialize inputs and output variables
    input_var = T.imatrix('inputs')
    target_var = T.imatrix('targets')
    #construct neural network
    network = build_MLP(input_var, outputDimns)
    #prediction function for training
    prediction = lasagne.layers.get_output(network, deterministic=False)
    #cost function for training
    loss = lasagne.objectives.squared_error(prediction, target_var)
    cost = loss.mean()
    #fetch parameter function
    all_params = lasagne.layers.get_all_params(network, trainable=True)
    #update function
    updates = lasagne.updates.nesterov_momentum(loss, all_params, 
                                                learning_rate=lr, 
                                                momentum=momentum)
    #prediction function for monitoring (validation)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    #cost function for monitoring (validation)
    test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
    test_cost = test_loss.mean()
    #compile functions
    train_fn = theano.function([input_var, target_var], [test_cost])
    val_fn = theano.function([input_var, target_var], [test_cost])
    print("Training started...")
    for epoch in xrange(num_epochs):
        #initialize error, batch no.s, and time
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, Y_train, inputDimns, 
                                         outputDimns, batch_size, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, Y_val, inputDimns, 
                                         outputDimns, batch_size, shuffle=False):
            inputs, targets = batch
            err = val_fn(inputs, targets)
            val_err += err
            val_batches += 1
        print("Epoch {} of {} took {:.3f}s".format(
            epoch+1, num_epochs, time.time() - start_time))
        print("training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, Y_test, inputDimns, 
                                     outputDimns, batch_size, 
                                     shuffle=False):
        inputs, targets = batch
        err = val_fn(inputs, targets)
        test_err += err
        test_batches += 1
    print("Final results:")
    print("Test Loss:\t\t\t{:.6f}".format(test_err/test_batches))


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        print("       input dropout and DROP_HID hidden dropout,")
        print("       'rnn' for a Recurrent Neural Network (RNN).")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else: 
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
            main(**kwargs)
