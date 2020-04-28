import numpy as np
import matplotlib.pylab as plt



def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c) #オーバーフロー対策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


a = np.array([0.3, 2.9, 4.8])
#a = np.array([10, 9, 8])
y = softmax(a)
print(y)
print(np.sum(y))