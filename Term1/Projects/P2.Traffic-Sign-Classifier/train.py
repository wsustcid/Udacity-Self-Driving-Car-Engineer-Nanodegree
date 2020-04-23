'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-04-21 10:28:37
@LastEditTime: 2020-04-23 11:12:21
'''
""" Use LeNet-5 to classify German Traffic Sign Dataset.
Goal:
0. 使用原始LeNet,达到0.89左右
1. The validation set accuracy will need to be at least 0.93. 

Example: http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf

调试过程：
1. 使用原始LeNet, 训练10个epoch左右，train_acc=1.0, val_acc=0.9 过拟合
2. LeNetv1: 去掉 一个fc4, val_acc=0.924, 仍然过拟合; 尝试去掉fc3, 0.91效果变差，此思路终止；
3. LeNetv2: 增加卷积层(无pool)：0.91; 增加卷积+pool: 0.80， 学习过程变慢，说明不适合更多的卷积特别是池化！

4. 说明改变结构已无用，需进行数据增强或大改网络结构

5. LeNetv1: lr=0.005, val_acc=0.938; 0.941(每次运行存在一定的随机性)
   将lr从0.005 改为0.0001,学习过程大大减慢，60个epoch才0.88；
   将lr=0.01, 跳动较大，但也到了0.936
   lr = 0.001, 0.91

6. 如果是真的打比赛，最好还是使用网格搜索，将所有参数组合尝试一遍
"""
import os 

import tensorflow as tf
from sklearn.utils import shuffle

from data_gen import load_data
from model.lenet import LeNetv1, LeNetv2



## Load data
training_file = './traffic-signs-data/train.p'
validation_file = './traffic-signs-data/valid.p'

X_train, y_train = load_data(training_file, process=True)
X_valid, y_valid = load_data(validation_file, process=True)


## Model
x = tf.placeholder(tf.float32, shape=(None,32,32,1))
y = tf.placeholder(tf.int32, shape=(None))
one_hot_y = tf.one_hot(y, 43)

logits = LeNetv1(x)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y,
                                                        logits=logits)
loss = tf.reduce_mean(cross_entropy)

## Train
EPOCHS = 20
BATCH_SIZE = 128
LR = 0.005
display_step=1

train_op = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss)

## Metrics
correct = tf.equal(tf.argmax(logits, 1), 
                   tf.argmax(one_hot_y, 1))
acc_op = tf.reduce_mean(tf.cast(correct, tf.float32))

## Model Saver
model_dir = './output/model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
save_file = os.path.join(model_dir, 'model.ckpt')
saver = tf.train.Saver(max_to_keep=1) # 仅保存最新的模型

## Training 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    n_samples = len(X_train)
    best_acc = 0.0
    
    print("Start Training...")
    for epochs_i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)

        for i in range(n_samples//BATCH_SIZE +1):
            # 若超出索引范围，会自动输出至末尾
            X_batch = X_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            y_batch = y_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            
            _, train_batch_acc = sess.run([train_op, acc_op], 
                                          feed_dict={x:X_batch, y:y_batch})
        
        ## Validation
        val_total_acc = 0.0
        for i in range(len(X_valid)//BATCH_SIZE +1):
            X_batch = X_valid[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            y_batch = y_valid[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            
            val_batch_loss, val_batch_acc=sess.run([loss, acc_op],feed_dict={x: X_batch, y:y_batch})
            val_total_acc += val_batch_acc*len(X_batch)

        val_acc = val_total_acc / len(X_valid)
        
        print('Epoch: %d, train_acc: %f, val_loss:%f, val_acc:%f'%(epochs_i+1,train_batch_acc, val_batch_loss, val_acc))

        ## save best model
        if val_acc > best_acc:
            best_acc = val_acc
            saver.save(sess, save_file, global_step=epochs_i+1)
            print("Best Model saved in %s", save_file)
        
    print("Training is done !")