# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 21:53:17 2020

@author: magic
"""

import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True,one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 100000
train_size = x_train.shape[0]
batch_size = 100
lerning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    #誤差逆伝搬によって勾配を求める
    grad = network.gradient(x_batch, t_batch)
    
    #更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= lerning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)        #配列に追加する
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(i, train_acc, test_acc, loss)
        
        
# グラフの描画
markers = {'train': 'o', 'test': 's'}
#x = np.arange(len(train_acc_list))
x = np.arange(len(train_loss_list))
#plt.plot(x, train_acc_list, label='train acc')
#plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.plot(x, train_loss_list, label='loss')
plt.xlabel("epochs")
#plt.ylabel("accuracy")
plt.ylabel("loss")
plt.ylim(0, 5.0)
plt.legend(loc='lower right')
plt.show()