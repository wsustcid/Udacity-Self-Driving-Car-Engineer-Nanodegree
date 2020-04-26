'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-04-10 21:06:45
@LastEditTime: 2020-04-26 09:32:54
'''
"""
使用AlexNet做特征提取，构建新的分类层，并进行fine tune
"""
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from model.alexnet import AlexNet

# TODO: Load traffic signs data.
with open('./dataset/train.p', 'rb') as f:
    train_data = pickle.load(f)


# TODO: Split data into training and validation sets.
features, labels = train_data['features'], train_data['labels']
features = (features-127.5)/127.5

X_train, X_valid, y_train, y_valid = train_test_split(features, labels,
                                                      test_size=0.2)
print("Train samples: {}".format(X_train.shape[0]))
print("Valid samples: {}".format(X_valid.shape[0]))

n_samples = len(X_train)
n_class = np.max(y_valid) + 1

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32,32,3))
x_resize = tf.image.resize_images(x,(227,227))
y = tf.placeholder(tf.int64, (None,))
y_one_hot = tf.one_hot(y,n_class)

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(x_resize, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
logits_w = tf.Variable(tf.truncated_normal(shape=(fc7.get_shape().as_list()[-1], n_class), stddev=1e-2, dtype=tf.float32))
logits_b = tf.Variable(tf.zeros(n_class))
logits = tf.nn.bias_add(tf.matmul(fc7, logits_w), logits_b)

# TODO: Define loss, training, accuracy operations.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_one_hot)
loss = tf.reduce_mean(cross_entropy)

correct = tf.equal(tf.argmax(logits, 1), y)
acc_op = tf.reduce_mean(tf.cast(correct, tf.float32))

# TODO: Train and evaluate the feature extraction model.
lr = 0.001
EPOCHS = 3
BATCH_SIZE = 32
# Note the var_list !!!
#Optional list or tuple of tf.Variable to update to minimize loss. Defaults to the list of variables collected in the graph under the key GraphKeys.TRAINABLE_VARIABLES.
# 指定了var_list, 只会计算并更新此列表中的变量；
# 如果没有指定，则默认都会计算梯度，但反向传播时会被tf.stop_gradient截止
# 因此二者使用其一即可
train_op = tf.train.AdamOptimizer(lr).minimize(loss, var_list=[logits_w,logits_b])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    print("=== Start Training ===")
    for epoch_i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        total_loss, total_acc = 0.0, 0.0
        n_batches = n_samples//BATCH_SIZE
        for i in range(n_batches):
            start = i*BATCH_SIZE
            end = (i+1)*BATCH_SIZE
            X_batch, y_batch = X_train[start:end], y_train[start:end]
            
            # 注意最重要的就是run train_op !!!
            _, loss_train, acc_train = sess.run([train_op, loss, acc_op], feed_dict={x:X_batch, y:y_batch})
            
            total_loss += loss_train
            total_acc += acc_train
        print("Epoch {}: train_loss: {}, train_acc: {}".format(epoch_i+1, total_loss/n_batches, total_acc/n_batches))

        total_loss, total_acc = 0.0, 0.0
        n_batches = len(X_valid)//BATCH_SIZE
        for i in range(n_batches):
            start = i*BATCH_SIZE
            end = (i+1)*BATCH_SIZE
            X_batch, y_batch = X_valid[start:end], y_valid[start:end]
            
            loss_train, acc_train = sess.run([loss,acc_op], feed_dict={x:X_batch, y:y_batch})
            
            total_loss += loss_train
            total_acc += acc_train
        
        print("valid_loss: {}, valid_acc: {}".format(total_loss/n_batches, total_acc/n_batches))
    
    print("==== Training done ! ===")

        
        
"""
仅使用stop_gradient:
Epoch 1: train_loss: 2.209506271688306, train_acc: 0.3934948979591837
valid_loss: 1.6952909537724086, valid_acc: 0.5283163265306122
Epoch 2: train_loss: 1.5654380302039945, train_acc: 0.5598533163265306
valid_loss: 1.4245986191593871, valid_acc: 0.5983418367346939

仅使用var_list:
=== Start Training ===
Epoch 1: train_loss: 2.2111520191844627, train_acc: 0.39419642857142856
valid_loss: 1.757074648993356, valid_acc: 0.514030612244898
Epoch 2: train_loss: 1.5656912539686476, train_acc: 0.5628826530612245
valid_loss: 1.409837415996863, valid_acc: 0.5963010204081632
Epoch 3: train_loss: 1.325853159780405, train_acc: 0.6308673469387756
valid_loss: 1.2584393885670877, valid_acc: 0.6545918367346939

二者配合使用：

.cc:1084] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1225 MB memory) -> physical GPU (device: 0, name: GeForce 940MX, pci bus id: 0000:03:00.0, compute capability: 5.0)
=== Start Training ===
Epoch 1: train_loss: 2.196659608641449, train_acc: 0.4009247448979592
valid_loss: 1.7509871312550136, valid_acc: 0.5089285714285714
Epoch 2: train_loss: 1.552603174593984, train_acc: 0.5650829081632653
valid_loss: 1.4908907442676778, valid_acc: 0.5831632653061225
Epoch 3: train_loss: 1.316674475341427, train_acc: 0.6317602040816327
valid_loss: 1.3422525145569626, valid_acc: 0.6169642857142857
Epoch 4: train_loss: 1.168740680448863, train_acc: 0.6744579081632653
valid_loss: 1.1438591937629543, valid_acc: 0.6883928571428571
Epoch 5: train_loss: 1.0585126057571295, train_acc: 0.7051339285714285
valid_loss: 1.047569277091902, valid_acc: 0.7071428571428572
Epoch 6: train_loss: 0.9774833819695882, train_acc: 0.7277104591836735
valid_loss: 0.9830967090567764, valid_acc: 0.7237244897959184
Epoch 7: train_loss: 0.9062546037593666, train_acc: 0.7480548469387756
valid_loss: 0.9337884681565422, valid_acc: 0.7309948979591837
Epoch 8: train_loss: 0.8514303081801959, train_acc: 0.7625318877551021
valid_loss: 0.8921746363445204, valid_acc: 0.7488520408163265
Epoch 9: train_loss: 0.8054263041791867, train_acc: 0.7737882653061224
valid_loss: 0.8348473192477713, valid_acc: 0.7686224489795919
Epoch 10: train_loss: 0.7662785344890186, train_acc: 0.785905612244898
valid_loss: 0.8235840278012412, valid_acc: 0.7489795918367347
Epoch 11: train_loss: 0.7274538353085518, train_acc: 0.8004783163265307
valid_loss: 0.7479820565301545, valid_acc: 0.7947704081632653
Epoch 12: train_loss: 0.6924139141732333, train_acc: 0.8102359693877551
valid_loss: 0.7543218255043029, valid_acc: 0.7831632653061225
Epoch 13: train_loss: 0.6646437636443547, train_acc: 0.8189413265306122
valid_loss: 0.6869811354851236, valid_acc: 0.8140306122448979
Epoch 14: train_loss: 0.6383141125191231, train_acc: 0.8243941326530613
valid_loss: 0.6822113932395468, valid_acc: 0.7951530612244898
Epoch 15: train_loss: 0.6118904454519554, train_acc: 0.8311543367346939
valid_loss: 0.6492827151502881, valid_acc: 0.8188775510204082
Epoch 16: train_loss: 0.5920222611907794, train_acc: 0.8380102040816326
valid_loss: 0.6347712179835961, valid_acc: 0.8237244897959184
Epoch 17: train_loss: 0.5713512651774348, train_acc: 0.8408801020408163
valid_loss: 0.6224151577268328, valid_acc: 0.8298469387755102
Epoch 18: train_loss: 0.5517095318254159, train_acc: 0.8491709183673469
valid_loss: 0.6046791454967188, valid_acc: 0.829719387755102
Epoch 19: train_loss: 0.5338735472638996, train_acc: 0.8554209183673469
valid_loss: 0.581237553698676, valid_acc: 0.8375
Epoch 20: train_loss: 0.5190219332825164, train_acc: 0.858545918367347
valid_loss: 0.5546787218171723, valid_acc: 0.8489795918367347
"""

