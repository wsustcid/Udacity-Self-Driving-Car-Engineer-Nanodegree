'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-04-03 10:11:42
@LastEditTime: 2020-04-07 08:37:18
'''
# Quiz Solution
# Note: You can't run code in this tab
import tensorflow as tf


def run():
    output = None
    x = tf.placeholder(tf.int32)

    with tf.Session() as sess:
        output = sess.run(x, feed_dict={x: 123})
        print output
    return output

# Solution is available in the other "solution.py" tab
import tensorflow as tf

# TODO: Convert the following to TensorFlow:
x = tf.constant(10.)
y =tf.constant(2.)
z =tf.subtract(tf.divide(x,y),tf.constant(1.))

# TODO: Print z from a session
with tf.Session() as sees:
    output=sees.run(z)
    print (output)
