'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-04-10 21:06:45
@LastEditTime: 2020-04-26 09:31:09
'''

"""
The traffic signs are 32x32 so you
have to resize them to be 227x227 before
passing them to AlexNet.
"""

import time
import tensorflow as tf
import numpy as np
from imageio import imread
from dataset.caffe_classes import class_names
from model.alexnet import AlexNet

"""
直接使用原始AlexNet进行交通标识的推理（也是用原始标签，1000类）
结果无任何匹配度，因为数据集标签不同
"""
# create a (32,32,3) tensor for traffic sign image 
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
# TODO: Resize the images so they can be fed into AlexNet.
# HINT: Use `tf.image.resize_images` to resize the images
resized = tf.image.resize_images(x, size=(227,227)) # size=(n_h,n_w)
assert resized is not Ellipsis, "resized needs to modify the placeholder image size to (227,227)"


probs = AlexNet(resized)

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
        print("%s: %.3f" % (class_names[inds[-1 - i]], output[input_im_ind, inds[-1 - i]]))
    print()

print("Time: %.3f seconds" % (time.time() - t))

# Image 0
'''
screen, CRT screen: 0.051
digital clock: 0.041
laptop, laptop computer: 0.030
balance beam, beam: 0.027
parallel bars, bars: 0.023

Image 1
digital watch: 0.395
digital clock: 0.275
bottlecap: 0.115
stopwatch, stop watch: 0.104
combination lock: 0.086

Time: 0.592 seconds
'''