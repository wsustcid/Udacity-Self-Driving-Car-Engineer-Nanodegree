'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-04-23 15:35:07
@LastEditTime: 2020-04-23 16:47:24
'''

""" Visualize the Neural Network's State with Test Images
Background:
  1. While neural networks can be a great learning device, they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. 
  2. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. 

Reference:
[End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) 
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from data_gen import load_data, load_sign_names

def outputFeatureMap(sess, image_input, tf_activation, activation_min=-1, activation_max=-1, plt_num=1):
    """
    Params:
    - image_input: the test image being fed into the network to produce the feature maps
    - tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
    - activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
    - plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry
    
    """
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})

    featuremaps = activation.shape[3]

    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 and activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")

        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")

    plt.show()





if __name__ == "__main__":
    ## Load test data
    testing_file = './traffic-signs-data/test.p'
    X_test,  y_test  = load_data(testing_file, process=True)
    sign_name_file = './traffic-signs-data/signnames.csv'
    sign_name = load_sign_names(sign_name_file)
    
    model_file = tf.train.latest_checkpoint('./output/model/')
    
    random_index = np.random.randint(0,len(X_test),size=1) # 产生的是一个一维数组；

    visual_image = X_test[random_index] # 即使取一张图片，使用数组索引其维度也不会改变
    print(visual_image.shape)
    

    ## Model
    x = tf.placeholder(tf.float32, shape=(None,32,32,1))
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

    # 后面不用的的层可以不用定义
        
    # save model (必须在此之前创建模型)
    saver = tf.train.Saver()

    ## Eval
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    
    # Restore model
    saver.restore(sess, model_file)

    outputFeatureMap(sess, visual_image, 
                    tf_activation=conv2)

    
    sess.close
