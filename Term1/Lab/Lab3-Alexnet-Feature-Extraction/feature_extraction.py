'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-04-10 21:06:45
@LastEditTime: 2020-04-26 09:32:39
'''
"""
使用AlexNet做特征提取，构建新的分类层
"""

import time
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib.pyplot import imread
from model.alexnet import AlexNet

sign_names = pd.read_csv('./dataset/signnames.csv')
nb_classes = 43

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, (227, 227))

# return the second to last layer 
fc7 = AlexNet(resized, feature_extract=True)
# TODO: Define a new fully connected layer 
# followed by a softmax activation to classify
shape = (fc7.get_shape().as_list()[-1], nb_classes)
logits_w = tf.Variable(tf.truncated_normal(shape=shape, stddev=1e-2, dtype=tf.float32)) # 一定要指定stddev
logits_b = tf.Variable(tf.zeros(nb_classes))
logits = tf.matmul(fc7,logits_w) + logits_b

probs = tf.nn.softmax(logits)

## 
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Read Images
im1 = imread("./test_img/construction.jpg").astype(np.float32)
im1 = im1 - np.mean(im1)

im2 = imread("./test_img/stop.jpg").astype(np.float32)
im2 = im2 - np.mean(im2)

# Run Inference
t = time.time()
output = sess.run(probs, feed_dict={x: [im1, im2]})

# Print Output
for input_im_ind in range(output.shape[0]):
    inds = np.argsort(output)[input_im_ind, :]
    print("Image", input_im_ind)
    for i in range(5):
        print("%s: %.3f" % (sign_names.iloc[inds[-1 - i]][1], output[input_im_ind, inds[-1 - i]]))
    print()

print("Time: %.3f seconds" % (time.time() - t))
