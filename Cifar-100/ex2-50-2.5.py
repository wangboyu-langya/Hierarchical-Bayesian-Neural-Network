# -*- coding: utf-8 -*-
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from Train import Train

# run on screen exp3
f = 8
fact = 2.5
name = 'ex2-50-2.5'
data_pth = '/home/hxianglong' + '/Data/cifar-100-python-cnn-features-transform-order'
lr = 7e-2
a = 10e1 * f
batch_size = 256 / f
epochs = 50
num_cls = 100
train_cls = 100
train_pr_cl = 500 + 15 * (num_cls - 1)
test_pr_cl = 100
print_freq = 10
train = Train(name, data_pth, batch_size, epochs, num_cls, train_cls, test_pr_cl, train_pr_cl, print_freq=print_freq, a=a, learning_rate=lr, factor=fact)
train.train()
