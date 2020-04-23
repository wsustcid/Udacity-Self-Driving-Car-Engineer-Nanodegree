'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-04-21 12:14:31
@LastEditTime: 2020-04-23 00:04:22
'''



import tensorflow as tf
from tensorflow.contrib.layers import flatten


def LeNet(x): 
    """
    Input: (32,32,1)
    output: 43
    """
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

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    fc5_w = tf.Variable(tf.truncated_normal(shape=(84,43),
                                            dtype=tf.float32,
                                            mean=mu, stddev=sigma))
    fc5_b = tf.Variable(tf.zeros(43, dtype=tf.float32))
    logits = tf.add(tf.matmul(fc4, fc5_w), fc5_b)
    
    return logits

def LeNetv1(x): 
    """
    Input: (32,32,1)
    output: 43
    """
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
    
    # Layer 5: Fully Connected. Input = 84. Output = 43.
    fc5_w = tf.Variable(tf.truncated_normal(shape=(120,43),
                                            dtype=tf.float32,
                                            mean=mu, stddev=sigma))
    fc5_b = tf.Variable(tf.zeros(43, dtype=tf.float32))
    logits = tf.add(tf.matmul(fc3, fc5_w), fc5_b)
    
    return logits



def LeNetv2(x): 
    """
    Input: (32,32,1)
    output: 43
    """
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
    
    # Layer 2: Convolutional. Output = 3x3x32.
    conv3_w = tf.Variable(tf.truncated_normal(shape=(3,3,16,32), 
                                              dtype=tf.float32,
                                              mean=mu, stddev=sigma))
    conv3_b = tf.Variable(tf.zeros(32, dtype=tf.float32))
    conv3 = tf.nn.conv2d(conv2_pool, conv3_w, strides=[1,1,1,1],
                         padding='VALID')
    conv3 = tf.nn.bias_add(conv3, conv3_b)
    conv3 = tf.nn.relu(conv3)
    
    # Pooling. Input = 3x3x32. Output = 1x1x32.
    conv3_pool = tf.nn.max_pool(conv3, [1,3,3,1], strides=[1,1,1,1],
                                padding='VALID')

    # Flatten. Input = 3x3x32. Output = 288.
    conv3_flatten = flatten(conv3_pool)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    fc5_w = tf.Variable(tf.truncated_normal(shape=(32,43),
                                            dtype=tf.float32,
                                            mean=mu, stddev=sigma))
    fc5_b = tf.Variable(tf.zeros(43, dtype=tf.float32))
    logits = tf.add(tf.matmul(conv3_flatten, fc5_w), fc5_b)
    
    return logits