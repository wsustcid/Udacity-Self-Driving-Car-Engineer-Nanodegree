'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-04-07 20:44:10
@LastEditTime: 2020-04-08 16:20:52
'''
# Load pickled data
import pickle
import numpy as np
#import tensorflow as tf
#tf.python.control_flow_ops = tf

with open('./small_traffic_set/small_train_traffic.p', mode='rb') as f:
    data = pickle.load(f)

X_train, y_train = data['features'], data['labels']

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten

# TODO: Build the Fully Connected Neural Network in Keras Here
# 128 hidden units and 5 outputs
model = Sequential()
model.add(Flatten(input_shape=(32,32,3)))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('softmax'))

# preprocess data
X_normalized = np.array(X_train / 255.0 - 0.5 )

from sklearn.preprocessing import LabelBinarizer

label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# TODO: change the number of training epochs to 3
history = model.fit(X_normalized, y_one_hot, nb_epoch=3,
                    validation_split=0.2)