
import sys, os
from dataset.mnist import load_mnist
import numpy as np
import matplotlib.pylab as plt
from PIL import Image
import pickle




def mean_squared_error(y,t):
    return 0.5 * np.sum((y-t)**2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)

    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) / batch_size



if __name__ == "__main__":
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=True)

    print(x_train.shape)
    print(t_train.shape)

    train_size = x_train.shape[0]
    batch_size = 10
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    print(batch_mask)

    """
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    #y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]

    print(cross_entropy_error(np.array(y), np.array(t)))
    """