# -*- coding: utf-8 -*-
import os, inspect, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from Train import Train
from Mail import mail
import traceback

factor = 1
f = 1
name = os.path.basename(__file__).split('.')[0]
data_pth = '/home/hxianglong' + '/Data/cifar-100-python-cnn-features-transform-order-no-neg'
batch_size = 256 * factor
epochs = 100
num_net = 10 / f
# train_net = 10 / f
train_net = 10
out_cls = 10 * f
train_pr_net = 5000 * f
test_pr_net = 1000 * f
print_freq = 5
val_freq = 20
a = 10e1
b = 10e1
c = 10e2
lr_func = 0
# para_ls = [(7e-2, -0.55), (7e-2, -0.54), (7e-2, -0.56), (7e-2, -0.525), (7e-2, -0.5), (7e-2, -0.575), (7e-2, -0.6), (6e-2, -0.55), (6e-2, -0.525), (6e-2, -0.5), (6e-2, -0.575), (6e-2, -0.6)]
para_ls = [(7e-2, -0.55), (7e-2, -0.54), (7e-2, -0.56), (7e-2, -0.525), (7e-2, -0.5), (7e-2, -0.575), (7e-2, -0.6)]
# para_ls = [(1.5e-2, 0.1), (1.25e-2, 0.1), (1.75e-2, 0.1), (2e-2, 0.1)]
# para_ls = [(1e-2, -0.4), (1e-2, -0.6), (1e-2, -0.7), (7e-2, -0.8), (5e-2, -0.55), (4e-2, -0.55), (4e-2, -0.45)]
train = Train(name=name, data_pth=data_pth, batch_size=batch_size, epochs=epochs, num_net=num_net, train_net=train_net,
              test_pr_net=test_pr_net, train_pr_net=train_pr_net, out_cls=out_cls, print_freq=print_freq,
              para_ls=para_ls, a=a, b=b, c=c, lr_func = lr_func) 
try:
    train.train(val_freq=val_freq, dir_txt=currentdir+'/')
except:
    e = traceback.format_exc()
    traceback.print_exc()
    info = 'ERROR in your code!!!\n\n %s'%e
    mail(subject=name + ' Exception', info=info)

