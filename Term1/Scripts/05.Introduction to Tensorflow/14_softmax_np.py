'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-04-03 10:11:40
@LastEditTime: 2020-04-07 09:46:50
'''

# Solution is available in the other "solution.py" tab
import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    # TODO: Compute and return softmax(x)
    return np.exp(x)/np.sum(np.exp(x), axis=0) # 1维向量只有0轴，二维矩阵零轴就是纵轴

logits = [3.0, 1.0, 0.2]
print(softmax(logits))
