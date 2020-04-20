'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-04-19 22:40:12
@LastEditTime: 2020-04-20 20:45:29
'''

""" 
Goal:
  1. In this lab, we'll use all the tools we learned from Class-5: Introduction to TensorFlow to label images of English letters! 
  2. The data we are using, notMNIST: <"http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html">, consists of images of a letter from A to J in differents font.
  3. A few examples of the data can be found at 'image/notmnist.png'. 
  4. Our goal is to make predictions against that test set with at least an 80% accuracy.

Dataset:
  1. The notMNIST dataset contains 500,000 images for just training.
  2. We will use a subset of this data, 15,000 images for each label (A-J).



"""


# Load the modules
import pickle
import math

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

## Load the data
pickle_file = './dataset/notMNIST.pickle'
with open(pickle_file, 'rb') as f:
  pickle_data = pickle.load(f)
  train_features = pickle_data['train_dataset']
  train_labels = pickle_data['train_labels']
  valid_features = pickle_data['valid_dataset']
  valid_labels = pickle_data['valid_labels']
  test_features = pickle_data['test_dataset']
  test_labels = pickle_data['test_labels']
  
  # Free up memory
  del pickle_data  

print('Data and modules loaded.')


## Create Tensor
features_count = 784
labels_count = 10

features = tf.placeholder(tf.float32, shape=(None, features_count))
labels = tf.placeholder(tf.float32, shape=(None, labels_count))

weights = tf.Variable(tf.truncated_normal(shape=(features_count, labels_count), dtype=tf.float32))
biases = tf.Variable(tf.zeros(shape=labels_count, dtype=tf.float32))


## Model: Linear Function WX + b
logits = tf.matmul(features, weights) + biases

prediction = tf.nn.softmax(logits)

## Cross entropy
cross_entropy = -tf.reduce_sum(labels * tf.log(prediction), axis=1)

## Training loss
loss = tf.reduce_mean(cross_entropy)

# Feed dicts for training, validation, and test session
train_feed_dict = {features: train_features, labels: train_labels}
valid_feed_dict = {features: valid_features, labels: valid_labels}
test_feed_dict = {features: test_features, labels: test_labels}

## Create an operation that initializes all variables
init = tf.global_variables_initializer()


## Accuracy
is_correct_prediction = tf.equal(tf.argmax(prediction, 1), 
                                 tf.argmax(labels, 1))

accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))


## Train
epochs = 10
batch_size = 200
learning_rate = 0.05


# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)    

# The accuracy measured against the validation set
validation_accuracy = 0.0
test_accuracy = 0.0

# Measurements use for graphing loss and accuracy
log_batch_step = 50
batches = []
loss_batch = []
train_acc_batch = []
valid_acc_batch = []

with tf.Session() as session:
    session.run(init)
    batch_count = int(math.ceil(len(train_features)/batch_size))

    for epoch_i in range(epochs):
        
        # Progress bar
        batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i+1, epochs), unit='batches')
        
        # The training cycle
        for batch_i in batches_pbar:
            # Get a batch of training features and labels
            batch_start = batch_i*batch_size
            batch_features = train_features[batch_start:batch_start + batch_size]
            batch_labels = train_labels[batch_start:batch_start + batch_size]

            # Run optimizer and get loss
            _, l = session.run(
                [optimizer, loss],
                feed_dict={features: batch_features, labels: batch_labels})

            # Log every 50 batches
            if not batch_i % log_batch_step:
                # Calculate Training and Validation accuracy
                training_accuracy = session.run(accuracy, feed_dict=train_feed_dict)
                validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

                # Log batches
                previous_batch = batches[-1] if batches else 0
                batches.append(log_batch_step + previous_batch)
                loss_batch.append(l)
                train_acc_batch.append(training_accuracy)
                valid_acc_batch.append(validation_accuracy)

        # Check accuracy against Validation data
        validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)
        print('Validation accuracy at {}'.format(validation_accuracy))

    
    # Check accuracy against Test data
    test_accuracy = session.run(accuracy, feed_dict=test_feed_dict)
    #assert test_accuracy >= 0.80, 'Test accuracy at {}, should be equal to or greater than 0.80'.format(test_accuracy)
    print('Nice Job! Test Accuracy is {}'.format(test_accuracy))

loss_plot = plt.subplot(211)
loss_plot.set_title('Loss')
loss_plot.plot(batches, loss_batch, 'g')
loss_plot.set_xlim([batches[0], batches[-1]])

acc_plot = plt.subplot(212)
acc_plot.set_title('Accuracy')
acc_plot.plot(batches, train_acc_batch, 'r', label='Training Accuracy')
acc_plot.plot(batches, valid_acc_batch, 'x', label='Validation Accuracy')
acc_plot.set_ylim([0, 1.0])
acc_plot.set_xlim([batches[0], batches[-1]])
acc_plot.legend(loc=4)

plt.tight_layout()
plt.show()

