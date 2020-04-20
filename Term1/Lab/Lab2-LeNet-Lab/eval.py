'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-04-21 00:32:57
@LastEditTime: 2020-04-21 00:51:28
'''


import tensorflow as tf

from data_gen import read_mnist
from lenet import LeNet

from sklearn.utils import shuffle

## Read data
_, _, _, _, X_test, y_test = read_mnist()

## Training Params
EPOCHS = 10
BATCH_SIZE = 128
lr = 0.001

## Inputs
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)


## Model
logits = LeNet(x)

## Loss and acc
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)

loss = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate = lr)
training_operation = optimizer.minimize(loss)



correct_prediction = tf.equal(tf.argmax(logits, 1), 
                              tf.argmax(one_hot_y, 1))

accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # Note the name!!

## Saver
save_file = './saved_model/train_model.ckpt'
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()

    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]

        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        
        total_accuracy += (accuracy * len(batch_x))
        
    return total_accuracy / num_examples

## Eval
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./saved_model'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))