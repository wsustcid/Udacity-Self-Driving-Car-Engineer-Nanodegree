'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-04-03 10:11:41
@LastEditTime: 2020-04-07 10:09:30
'''
# Solution is available in the other solution.py tab
import tensorflow as tf

softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)

# TODO Print cross entropy from session
cross_entropy=tf.reduce_sum(-tf.multiply(one_hot,tf.log(softmax)))
with tf.Session() as sees:
    print(sees.run(cross_entropy,feed_dict={softmax:softmax_data,one_hot:one_hot_data}))
    