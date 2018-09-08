# -*- coding: utf-8 -*-
import os,sys,inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0,parentdir)
from Train import Train
factor = 1
name = 'test-50'
data_pth = '/home/hxianglong' + '/Data/cifar-100-python-cnn-features-transform-order-no-neg'
batch_size = 256 * factor
epochs = 50
num_net = 10
train_net = 2
train_pr_net = 5000
test_pr_net = 1000
print_freq = 5
train = Train(name, data_pth, batch_size, epochs, num_net, train_net, test_pr_net, train_pr_net, print_freq=print_freq)
train.train()
