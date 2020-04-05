'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-04-03 10:10:41
@LastEditTime: 2020-04-05 11:39:18
'''

import numpy as np


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


x = np.array([0.5, 0.1, -0.2]) # 1x3
target = 0.6
learnrate = 0.5

weights_input_hidden = np.array([[0.5, -0.6],
                                 [0.1, -0.2],
                                 [0.1, 0.7]]) # 3x2

weights_hidden_output = np.array([0.1, -0.3]) # 1x2

## Forward pass
hidden_layer_input = np.dot(x, weights_input_hidden) # 1x2
hidden_layer_output = sigmoid(hidden_layer_input) # 1x2

output_layer_in = np.dot(hidden_layer_output, weights_hidden_output) # 1
output = sigmoid(output_layer_in) # 1

## Backwards pass
## TODO: Calculate error
error = target-output # 1

# TODO: Calculate error gradient for output layer
del_err_output = error*output*(1-output) # 1

# TODO: Calculate error gradient for hidden layer
del_err_hidden = weights_hidden_output*del_err_output*hidden_layer_output*(1-hidden_layer_output) # 1x2

# TODO: Calculate change in weights for hidden layer to output layer
delta_w_h_o = learnrate*del_err_output*hidden_layer_output # 1x2

# TODO: Calculate change in weights for input layer to hidden layer
delta_w_i_h = learnrate*del_err_hidden * x[:, None] # 1x2 * 3x1 => 3x2

print('Change in weights for hidden layer to output layer:')
print(delta_w_h_o)
print('Change in weights for input layer to hidden layer:')
print(delta_w_i_h)
