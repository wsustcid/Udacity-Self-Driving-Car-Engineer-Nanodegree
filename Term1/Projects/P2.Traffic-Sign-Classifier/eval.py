'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-04-23 09:47:49
@LastEditTime: 2020-04-23 16:11:05
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from data_gen import load_data, load_sign_names
from model.lenet import LeNetv1
 
def eval_batch(features,labels,model_file):
    """
    1. Restore model from ckpt
    2. Evaluate batch of data
    """
    ## Model
    x = tf.placeholder(tf.float32, shape=(None,32,32,1))
    y = tf.placeholder(tf.int32, shape=(None))
    one_hot_y = tf.one_hot(y, 43)

    logits = LeNetv1(x)
    
    correct = tf.equal(tf.argmax(logits, 1), 
                   tf.argmax(one_hot_y, 1))
    acc_op = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    ## Eval
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    
    # Restore model
    saver = tf.train.Saver()
    saver.restore(sess, model_file)

    test_acc = sess.run(acc_op, feed_dict={x:features,y:labels})
    print("Test acc: {}".format(test_acc))

    sess.close

def predict(features, model_file):
    """
    1. Restore model from ckpt
    2. Evaluate batch of data
    """
    ## Model
    x = tf.placeholder(tf.float32, shape=(None,32,32,1))
    logits = LeNetv1(x)
    scores = tf.nn.softmax(logits)
        
    ## Eval
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    
    # Restore model
    saver = tf.train.Saver()
    saver.restore(sess, model_file)

    top_k_values, top_k_indices = tf.nn.top_k(scores, k=3) # now top_k is tensor!!

    values, indices = sess.run([top_k_values, top_k_indices],
                               feed_dict={x:features}) 

    sess.close

    return values, indices


if __name__ == "__main__":
    ## Load test data
    testing_file = './traffic-signs-data/test.p'
    X_test,  y_test  = load_data(testing_file, process=True)
    sign_name_file = './traffic-signs-data/signnames.csv'
    sign_name = load_sign_names(sign_name_file)
    
    model_file = tf.train.latest_checkpoint('./output/model/')
    #eval_batch(X_test, y_test, model_file)
    
    random_index = np.random.randint(0,len(X_test),size=5)
    pred_images = X_test[random_index]
    top_k_value, top_k_indices = predict(pred_images,
                                         model_file)
                                     
    for i in range(len(pred_images)):
        print("Image {}, top_k_class: {}".format(i, np.array(sign_name)[top_k_indices[i]]))
        print("top_k_value: {}".format(top_k_value[i]))

        fig = plt.figure()
        plt.imshow(pred_images[i].reshape((32,32)), cmap='gray')
        plt.show()

