'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-04-20 23:13:05
@LastEditTime: 2020-04-21 00:26:00
'''

from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle
import numpy as np


def read_mnist():

    ## Load the MNIST data, which comes pre-loaded with TensorFlow.
    mnist = input_data.read_data_sets("MNIST_data/", reshape=False)

    X_train, y_train = mnist.train.images, mnist.train.labels
    X_valid, y_valid = mnist.validation.images, mnist.validation.labels
    X_test, y_test   = mnist.test.images, mnist.test.labels

    assert(len(X_train) == len(y_train))
    assert(len(X_valid) == len(y_valid))
    assert(len(X_test) == len(y_test))

    print()
    print("Image Shape: {}".format(X_train[0].shape))
    print()
    print("Training Set:   {} samples".format(len(X_train)))
    print("Validation Set: {} samples".format(len(X_valid)))
    print("Test Set:       {} samples".format(len(X_test)))

    ## Padding input images with 0
    # unique pad widths for each axis. ((before, after),) -> (N,H,W,C)
    X_train = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant') 
    X_valid = np.pad(X_valid, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    X_test  = np.pad(X_test,  ((0,0),(2,2),(2,2),(0,0)), 'constant')
        
    print("Padded Image Shape: {}".format(X_train[0].shape))

    ## Data processing
    # shuffle data
    X_train, y_train = shuffle(X_train, y_train)

    return X_train, y_train, X_valid, y_valid, X_test, y_test