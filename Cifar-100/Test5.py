# -*- coding: utf-8 -*-
import os,sys,inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0,parentdir)
from Train import Train
factor = 1
f = 2
name = 'test5'
data_pth = '/home/hxianglong' + '/Data/cifar-100-python-cnn-features-transform-order-no-neg'
batch_size = 256 * factor
epochs = 2
num_net = 10 / f
train_net = 10 / f
out_cls = 10 * f
train_pr_net = 5000 * f
test_pr_net = 1000 * f
print_freq = 5
train = Train(name, data_pth, batch_size, epochs, num_net, train_net, test_pr_net, train_pr_net, print_freq=print_freq)
train.train(out_cls=out_cls)
