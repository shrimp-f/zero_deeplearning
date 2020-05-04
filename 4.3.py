
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)


def function_1(x):
    return 0.01*x**2 + 0.1*x


def function_2(x):
    return x[0]**2 + x[1]**2


def main_1():
    x = np.arange(0.0, 20.0, 0.1)
    y = function_1(x)

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x, y)
    plt.show()

    print(numerical_diff(function_1, 5))


def main_2():
    x = [[],[]]
    x[0] = np.arange(-3.0, 3.0, 0.1)
    x[1] = np.arange(-3.0, 3.0, 0.1)

    X = np.meshgrid(x[0], x[1])
    Y = function_2(X)

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_zlabel("f(x0, x1)")

    ax.plot_wireframe(X[0], X[1], Y)
    plt.show()


if __name__ == "__main__":
    main_2()