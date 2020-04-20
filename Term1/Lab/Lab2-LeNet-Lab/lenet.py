'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-04-20 22:09:38
@LastEditTime: 2020-04-21 00:28:40
'''

"""
### Input
The LeNet architecture accepts a 32x32xC image as input, where C is the number of color channels. Since MNIST images are grayscale, C is 1 in this case.

### Architecture
**Layer 1: Convolutional.** The output shape should be 28x28x6.

**Activation.** Your choice of activation function.

**Pooling.** The output shape should be 14x14x6.

**Layer 2: Convolutional.** The output shape should be 10x10x16.

**Activation.** Your choice of activation function.

**Pooling.** The output shape should be 5x5x16.

**Flatten.** Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. The easiest way to do is by using `tf.contrib.layers.flatten`, which is already imported for you.

**Layer 3: Fully Connected.** This should have 120 outputs.

**Activation.** Your choice of activation function.

**Layer 4: Fully Connected.** This should have 84 outputs.

**Activation.** Your choice of activation function.

**Layer 5: Fully Connected (Logits).** This should have 10 outputs.

### Output
Return the result of the 2nd fully connected layer.

"""
import tensorflow as tf
from tensorflow.contrib.layers import flatten


def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_w = tf.Variable(tf.truncated_normal(shape=(5,5,1,6), 
                                              dtype=tf.float32,
                                              mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6, dtype=tf.float32))
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1,1,1,1], 
                         padding='VALID')
    conv1 = tf.nn.bias_add(conv1, conv1_b)
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1_pool = tf.nn.max_pool(conv1, [1,2,2,1], strides=[1,2,2,1],
                                padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_w = tf.Variable(tf.truncated_normal(shape=(5,5,6,16), 
                                              dtype=tf.float32,
                                              mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16, dtype=tf.float32))
    conv2 = tf.nn.conv2d(conv1_pool, conv2_w, strides=[1,1,1,1],
                         padding='VALID')
    conv2 = tf.nn.bias_add(conv2, conv2_b)
    conv2 = tf.nn.relu(conv2)
    
    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2_pool = tf.nn.max_pool(conv2, [1,2,2,1], strides=[1,2,2,1],
                                padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    conv2_flatten = flatten(conv2_pool)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc3_w = tf.Variable(tf.truncated_normal(shape=(400,120),
                                            dtype=tf.float32,
                                            mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(120, dtype=tf.float32))
    fc3 = tf.add(tf.matmul(conv2_flatten, fc3_w), fc3_b)
    fc3 = tf.nn.relu(fc3)
    
    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc4_w = tf.Variable(tf.truncated_normal(shape=(120,84),
                                            dtype=tf.float32,
                                            mean=mu, stddev=sigma))
    fc4_b = tf.Variable(tf.zeros(84, dtype=tf.float32))
    fc4 = tf.add(tf.matmul(fc3, fc4_w), fc4_b)
    fc4 = tf.nn.relu(fc4)

    # Layer 5: Fully Connected. Input = 84. Output = 10.
    fc5_w = tf.Variable(tf.truncated_normal(shape=(84,10),
                                            dtype=tf.float32,
                                            mean=mu, stddev=sigma))
    fc5_b = tf.Variable(tf.zeros(10, dtype=tf.float32))
    logits = tf.add(tf.matmul(fc4, fc5_w), fc5_b)
    
    return logits