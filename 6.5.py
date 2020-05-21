import sys, os
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.functions import *
from common.gradient import numerical_gradient
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD

from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer
from common.util import shuffle_dataset


if __name__ == "__main__":
#    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=True)
    (x_train, t_train), (x_test, t_test) = load_mnist()

    x_train, t_train = shuffle_dataset(x_train, t_train)

    validation_rate = 0.20
    validation_num = int(x_train.shape[0] * validation_rate)

    x_val = x_train[:validation_num]
    t_val = t_train[:validation_num]
    x_train = x_train[validation_num:]
    t_train = t_train[validation_num:]


    # ハイパーパラメータのランダム探索======================================
    optimization_trial = 3
    results_val = {}
    results_train = {}

    for _ in range(optimization_trial):
        weight_decay = 10 ** np.random.uniform(-3, 3)
        lr = 10 ** np.random.uniform(-6, -2)
        epochs_num=10
        
        network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10, weight_decay_lambda=weight_decay)
        trainer = Trainer(network, x_train, t_train, x_val, t_val,
                        epochs=epochs_num, mini_batch_size=100,
                        optimizer='sgd', optimizer_param={'lr': 0.01}, verbose=False)
        trainer.train()

        val_acc_list, train_acc_list = trainer.train_acc_list, trainer.test_acc_list

        print("val acc:" + str(val_acc_list[-1]) + " | lr:" + str(lr) + ", weight decay:" + str(weight_decay))
        key = "lr:" + str(lr) + ", weight decay:" + str(weight_decay)
        results_val[key] = val_acc_list
        results_train[key] = train_acc_list


    # グラフの描画========================================================
    print("=========== Hyper-Parameter Optimization Result ===========")
    graph_draw_num = 20
    col_num = 5
    row_num = int(np.ceil(graph_draw_num / col_num))
    i = 0

    for key, val_acc_list in sorted(results_val.items(), key=lambda x:x[1][-1], reverse=True):
        print("Best-" + str(i+1) + "(val acc:" + str(val_acc_list[-1]) + ") | " + key)

        plt.subplot(row_num, col_num, i+1)
        plt.title("Best-" + str(i+1))
        plt.ylim(0.0, 1.0)
        if i % 5: plt.yticks([])
        plt.xticks([])
        x = np.arange(len(val_acc_list))
        plt.plot(x, val_acc_list)
        plt.plot(x, results_train[key], "--")
        i += 1

        if i >= graph_draw_num:
            break

    plt.show()