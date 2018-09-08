# -*- coding: utf-8 -*-
import os,sys,inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0,parentdir)
from Train import Train
factor = 1
name = 'test-150-0.45'
data_pth = '/home/hxianglong' + '/Data/cifar-100-python-cnn-features-transform-order-no-neg'
batch_size = 256 * factor
epochs = 150
num_net = 10
train_net = 10
train_pr_net = 5000
test_pr_net = 1000
print_freq = 5
decay = -0.45
train = Train(name, data_pth, batch_size, epochs, num_net, train_net, test_pr_net, train_pr_net, print_freq=print_freq, decay=decay)
train.train()
