#同じディレクトリにあるdatasetディレクトリを使用
#元は、https://github.com/oreilly-japan/deep-learning-from-scratch　のdataset

import sys, os
from dataset.mnist import load_mnist
import numpy as np
import matplotlib.pylab as plt
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)