'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-04-03 10:11:23
@LastEditTime: 2020-04-06 18:28:45
'''
def gradient_descent_update(x, gradx, learning_rate):
    """
    Performs a gradient descent update.
    """
    # TODO: Implement gradient descent.
    x = x - learning_rate*gradx
    # Return the new value for x
    return x
import f