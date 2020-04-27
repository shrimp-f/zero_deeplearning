import numpy as np
import matplotlib.pylab as plt



def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c) #オーバーフロー対策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


a = np.array([1010, 1000, 990])
#a = np.array([10, 9, 8])
print(softmax(a))