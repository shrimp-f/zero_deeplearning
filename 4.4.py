import sys, os
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from dataset.mnist import load_mnist
from common.functions import softmax, cross_entropy_error
#from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


def function_2(x):
    return x[0]**2 + x[1]**2


def _numerical_gradient_no_batch(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        #f(x+h)の計算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        #f(x-h)の計算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
    
    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
        
        return grad


def gradient_dexcent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x



def main_1():
        x0 = np.arange(-2, 2.5, 0.25)
        x1 = np.arange(-2, 2.5, 0.25)
        X, Y = np.meshgrid(x0, x1)
        
        X = X.flatten()
        Y = Y.flatten()

        grad = numerical_gradient(function_2, np.array([X, Y]).T).T

        plt.figure()
        plt.quiver(X, Y, -grad[0], -grad[1],  angles="xy",color="#666666")
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.xlabel('x0')
        plt.ylabel('x1')
        plt.grid()
        plt.draw()
        plt.show()


def main_4_4_1():
    init_x = np.array([-3.0, 4.0])
    print(gradient_dexcent(function_2, init_x=init_x, lr=0.01, step_num=1000))




if __name__ == "__main__":
    net = simpleNet()
    print(net.W)

    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(p)

    print(np.argmax(p))

    t = np.array([0, 0, 1])
    print(net.loss(x, t))

    f = lambda w: net.loss(x, t)
    dW = numerical_gradient(f, net.W)
    print(dW)