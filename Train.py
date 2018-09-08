# -*- coding: utf-8 -*-
import matplotlib

matplotlib.use('Agg')

import shutil
import time
from datetime import timedelta
import math
import sys
import os, inspect

import numpy as np
import torch
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from AverageMeter import AverageMeter
from Net import Hbnn
from Mail import mail

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENTDIR = os.path.dirname(CURRENTDIR)
DEBUG = False


# DEBUG = True


# Done:  experiment on cifar-100
# TODO:  try to implement a multi-gpu version
class Train():
    # incrementally add training data class by class
    def __init__(self, name, data_pth, batch_size, epochs, num_net, train_net, test_pr_net, train_pr_net, out_cls,
                 para_ls,
                 num_features=456, minus=True, factor=1,
                 a=10e1, b=10e1, c=10e2, resume=False, print_freq=5,
                 learning_rate=7e-2,
                 shuffle=True, parallel=False, val_freq=4, decay=-0.55, lr_func=0, tp1=70, tp2=120):

        self.name = name
        self.data_pth = data_pth

        self.minus = minus
        self.factor = factor
        self.a = a
        self.b = b
        self.c = c

        self.num_net = num_net
        self.out_cls = out_cls
        self.train_net = train_net
        self.train_pr_net = train_pr_net
        self.test_pr_net = test_pr_net
        self.num_features = num_features
        self.batch_size = batch_size
        self.epochs = epochs
        self.print_freq = print_freq

        self.parallel = parallel
        self.if_resume = resume

        self.hbnn = Hbnn()
        self.elbo = list()

        # self.learning_rate = learning_rate
        # self.decay = decay
        self.para_ls = para_ls  # an iterator contains the parameters list to be experimented
        self.lr_func = lr_func


        self.turn_pt_1 = tp1
        self.turn_pt_2 = tp2

        print(('=> Experiment parameters:\n'
               'factor: [{0}]\t a: [{1}]\t b: [{2}]\t c: [{3}]\t'
               'batch: [{4}]\t epoch: [{6}]\t lr: [{5}]\n'
               'decay: [{8}]\t cls: [{7}]\t class batch: {9}').format(factor, a, b, c, batch_size, learning_rate,
                                                                      epochs, train_net, decay, out_cls))

    def __parameters(self, prec1_ls, best_prec1_all_ls, epoch, net, state_dict, elbos, prec1_train, fail, message,
                     code, seed):
        """save the model parameters in a dict"""
        parameters = {
            'name': self.name,
            'data_pth': self.data_pth,

            'minus': self.minus,
            'f': self.factor,
            'a': self.a,
            'b': self.b,
            'c': self.c,

            'num_net': self.num_net,
            'out_cls': self.out_cls,
            'train_net': self.train_net,
            'train_pr_net': self.train_pr_net,
            'test_pr_net': self.test_pr_net,
            'num_features': self.num_features,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'print_freq': self.print_freq,

            'parallel': self.parallel,
            'if_resume': self.if_resume,

            'hbnn': self.hbnn,

            'learning_rate': self.learning_rate,
            'decay': self.decay,
            # these are parameters specific to training
            'prec1_ls': prec1_ls,
            'best_prec1_all_ls': best_prec1_all_ls,
            'epoch': epoch,
            'net': net,
            'state_dict': state_dict,
            'elbos': elbos,
            'prec_train': prec1_train,
            'fail': fail,
            'message': message,
            'code': code,
            'seed': seed
        }
        return parameters

    def __load_data(self, path):
        """load the data"""
        print("=> loading data @ '{}'".format(path))
        x_train = np.load(path + '/features_train.npy')
        y_train = np.load(path + '/targets_train.npy')
        x_test = np.load(path + '/features_test.npy')
        y_test = np.load(path + '/targets_test.npy')
        print("=> loaded data @ '{}'".format(path))
        # reshape training data in class order
        x_train_ord = x_train.reshape(self.num_net, self.train_pr_net, self.num_features)
        y_train_ord = y_train.reshape(self.num_net, self.train_pr_net)
        x_test_ord = x_test.reshape(self.num_net, self.test_pr_net, self.num_features)
        y_test_ord = y_test.reshape(self.num_net, self.test_pr_net)

        return x_train_ord, y_train_ord, x_test_ord, y_test_ord, x_test, y_test

    def resume(self, pth, net_ls, dir_txt, val_freq=10, shuf=True, seed=0, start_epoch=0, epoch_trap=40):
        if os.path.isfile(pth):
            model = Hbnn()
            print("=> loading checkpoint '{}'".format(pth))
            cp = torch.load(pth)
            # start_epoch = checkpoint['epoch']
            # best_prec1_ls = checkpoint['best_prec1_ls']
            # cls = checkpoint['cls']
            # parameters = checkpoint['parameters']
            # self.__reset_params()
            for i in range(self.num_net):
                model.addnet()
            model.load_state_dict(cp['state_dict'])
            print("=> loaded checkpoint '{}'".format(pth))
            model.cuda()
            print('training class {}'.format(net_ls))

            directory = CURRENTDIR + "/runs/%s/imgs/" % (self.name)
            if not os.path.exists(directory):
                os.makedirs(directory)
            flag_1 = time.time()  # timing flag for whole trainning
            fail_cnt = 0
            self.start_epoch = start_epoch
            # prec1_ls = list()  # store the accuracy for every single net
            # best_prec1_ls = list()  # store the best accuracy for every single net
            # best_prec1_all_ls = list()  # store the best accuracy for the whole net
            # prec1_ls = list()  # store the accuracy for every single net
            # best_prec1_ls = list()  # store the best accuracy for every single net
            # best_prec1_all_ls = list()  # store the best accuracy for the whole net
            prec1_ls = np.zeros(self.train_net)  # store the accuracy for every single net
            best_prec1_ls = np.zeros(self.train_net) # store the best accuracy for every single net
            best_prec1_all_ls = np.zeros(self.train_net)  # store the best accuracy for the whole net
            elbos = np.zeros((self.train_net, self.epochs, 4))
            prec1_train = np.zeros((self.train_net, self.epochs))
            # model = Hbnn(self.out_cls)
            x_train_ord, y_train_ord, x_test_ord, y_test_ord, x_test, y_test = self.__load_data(self.data_pth)

            pngs = ['overall_preresume.png', 'single_preresume.png']
            prec_ls, prec_all_ls = np.zeros(self.num_net), np.zeros(self.num_net),
            for net in range(self.num_net):
                x_testing, y_testing = shuffle(x_test_ord[net], y_test_ord[net], random_state=seed)
                x_val, y_val = shuffle(x_test[0: (net + 1) * self.test_pr_net],
                                       y_test[0: (net + 1) * self.test_pr_net], random_state=seed)
                testing = data_utils.TensorDataset(torch.FloatTensor(x_testing), torch.IntTensor(y_testing))
                testing_loader = data_utils.DataLoader(testing, batch_size=self.batch_size, shuffle=shuf)
                val = data_utils.TensorDataset(torch.FloatTensor(x_val), torch.IntTensor(y_val))
                val_loader = data_utils.DataLoader(val, batch_size=self.batch_size, shuffle=shuf)
                prec = self.__validate_net(testing_loader, model, net)
                prec_all = self.validate(val_loader, model, net)
                prec_ls[net] = prec
                prec_all_ls[net] = prec_all
            self.plot(range(self.out_cls, (self.num_net + 1) * self.out_cls, self.out_cls), prec_all_ls, name=pngs[0],
                      caption='overall test accuracy')
            self.plot(range(self.out_cls, (self.num_net + 1) * self.out_cls, self.out_cls), prec_ls, name=pngs[1],
                      caption='single class accuracy')
            mail_body = (
                'Your training <<{0}>> has been tested before resume, time cost: {1}.\n'
            ).format(self.name, str(timedelta(seconds=(time.time() - flag_1))).split('.')[0])
            print(mail_body)
            mail(subject='%s preresume test' % self.name, pngs=pngs, dir_pic=directory,
                 txts=[self.name + '.txt'], dir_txt=dir_txt, info=mail_body)


            # if_resume the training
            for net in net_ls:
                flag_2 = time.time()  # timing for a single net

                # prepare the training data
                if shuf:
                    # data used to training
                    x_training, y_training = shuffle(x_train_ord[net], y_train_ord[net], random_state=seed)
                    # data used to test single class
                    x_testing, y_testing = shuffle(x_test_ord[net], y_test_ord[net], random_state=seed)
                    # data used to test the whole net
                    x_val, y_val = shuffle(x_test[0: (net + 1) * self.test_pr_net],
                                           y_test[0: (net + 1) * self.test_pr_net], random_state=seed)
                else:
                    x_training, y_training = x_train_ord[net], y_train_ord[net]
                    x_testing, y_testing = x_test_ord[net], y_test_ord[net]
                    x_val, y_val = x_test[0: (net + 1) * self.test_pr_net], y_test[0: (net + 1) * self.test_pr_net]
                # cut the data in minibatches and shuffle
                training = data_utils.TensorDataset(torch.FloatTensor(x_training), torch.IntTensor(y_training))
                training_loader = data_utils.DataLoader(training, batch_size=self.batch_size, shuffle=shuf)
                testing = data_utils.TensorDataset(torch.FloatTensor(x_testing), torch.IntTensor(y_testing))
                testing_loader = data_utils.DataLoader(testing, batch_size=self.batch_size, shuffle=shuf)
                val = data_utils.TensorDataset(torch.FloatTensor(x_val), torch.IntTensor(y_val))
                val_loader = data_utils.DataLoader(val, batch_size=self.batch_size, shuffle=shuf)

                # add a subnet for new class
                best_prec1 = 0  # best accuracy of a single net
                best_prec1_all = 0  # best accuracy for the whole network
                para_ls = iter(self.para_ls)
                self.learning_rate, self.decay = next(para_ls)
                # if self.lr_func == 0:
                #     self.learning_rate, self.decay = next(para_ls)
                # if self.lr_func == 1:
                #     self.learning_rate, self.decay = next(para_ls)
                # best_prec1_ls.append(best_prec1)
                # best_prec1_all_ls.append(best_prec1_all)

                # put the model on GPU, however, the multi-gpu is not conceivable at the moement
                if self.parallel:
                    model_paral = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
                else:
                    model.cuda()
                print('=> Number of trained parameters: {}'.format(
                    sum([p.data.nelement() for p in model.params_net(net)])))
                print('=> Total model parameters increased to: {}'.format(
                    sum([p.data.nelement() for p in model.parameters()])))

                # set the optimizer
                optimizer = torch.optim.RMSprop(model.params_net(net), lr=self.learning_rate, weight_decay=0.9)

                # segment of class
                print(' ' * 50 + ' Network ' + str(net) + ' ' + " " * 50)
                # for epoch in range(self.epochs):
                epoch = 0
                cnt = 0
                is_best_all = False
                prec1_all = 0
                best_prec1_all = 0
                while epoch < self.epochs:
                    lr = self.__adjust_lr(optimizer, epoch)
                    # train for one epoch
                    e = self.__train(training_loader, model, optimizer, net, epoch)
                    elbos[net, epoch] = np.asarray(e)
                    print('Learning Rate: [{0:.4f}]').format(lr)
                    # evaluate on validation set on a single class
                    prec1 = self.__validate_net(testing_loader, model, net, epoch)
                    prec1_train[net, epoch] = prec1
                    # evaluate on the single class
                    is_best = prec1 > best_prec1
                    best_prec1 = max(prec1, best_prec1)
                    if is_best:
                        best_prec1_ls[net] = best_prec1  # store the best prec

                    # evaluate on the whole net
                    if (epoch % val_freq == 0 and epoch != 0) or epoch == self.epochs - 1:
                        prec1_all = self.validate(val_loader, model, net, epoch)
                        # is_best_all = prec1_all > best_prec1_all
                        best_prec1_all = max(prec1_all, best_prec1_all)
                        if is_best_all:
                            best_prec1_all_ls[net] = best_prec1_all

                    # judge whether the training has failed
                    fail, message, code = self.__judge(net, epoch, self.epochs, prec1_train, elbos, epoch_trap=epoch_trap)

                    if fail:
                        filename = 'fail_{}_net_{}_epoch_{}'.format(fail_cnt, net, epoch)
                        self.__save_checkpoint({
                            'parameters': self.__parameters(prec1_ls, best_prec1_all_ls, epoch, net, model.state_dict(),
                                                            elbos, prec1_train, fail, message, code, seed)}, is_best, prec1,
                            net, filename)
                        # pngs = ['overall.png', 'single.png', 'prec_net_{}_train.png'.format(net), 'elbo_net_{}_train.png'.format(net)]
                        pngs = ['prec_net_{}_train_fail_{}.png'.format(net, fail_cnt),
                                'elbo_net_{}_train_fail_{}.png'.format(net, fail_cnt),
                                'lr_net_{}_fail_{}.png'.format(net, fail_cnt)]
                        self.plot(range(1, epoch + 2), prec1_train[net, 0:epoch + 1], name=pngs[0],
                                  caption='net %s train accuracy' % net, x_label='epoch')
                        self.plot(range(1, epoch + 2), elbos[net, 0:epoch + 1, 0], name=pngs[1],
                                  caption='net %s train elbo' % net,
                                  x_label='epoch', y_label='elbo')
                        self.__plot_lr(name=pngs[2])
                        fail_cnt += 1
                        cnt += 1
                        try:
                            # reset the parameters and train again
                            lr, dc = self.learning_rate, self.decay
                            self.learning_rate, self.decay = next(para_ls)
                            # model.reset(net)
                            # model.cuda()
                            model_tmp = Hbnn(self.out_cls)
                            for i in range(self.num_net):
                                model_tmp.addnet()
                            model_tmp.load_state_dict(cp['state_dict'])
                            model_tmp.cuda()
                            model.hbnn[net] = model_tmp.hbnn[net]
                            model.mu_gamma_g[net] = model_tmp.mu_gamma_g[net]
                            model.sigma_gamma_g[net] = model_tmp.sigma_gamma_g[net]
                            mail_body = (
                                'Your training <<{0}>> has failed {1}({12}) times in epoch {2}, net {3}, class batch {4}.\n'
                                'The model has been stored. The parameters have been adjusted and start to train this net again.\n'
                                'The reason is {5}, the exit code is {6}.\n'
                                'Total time this time for network {3}: {11}.\n'
                                'learning rate: [{7}] => [{8}]\t decay: [{9}] => [{10}]\n'
                            ).format(self.name, cnt, epoch, net, self.out_cls, message, code, lr, self.learning_rate,
                                     dc, self.decay, str(timedelta(seconds=(time.time() - flag_2))).split('.')[0], fail_cnt)
                            print(mail_body)
                            mail(subject='%s failed' % self.name, pngs=pngs,
                                 dir_pic=directory,
                                 txts=[self.name + '.txt'], dir_txt=dir_txt, info=mail_body)
                            epoch = 0
                            flag_2 = time.time()  # timing for a single net
                        except StopIteration:
                            mail_body = (
                                'Your training <<{0}>> has failed {1}({11}) times in epoch {2}, net {3}, class batch {4}.\n'
                                'The model has been stored. The parameters list has been exhausted and training has stopped.\n'
                                'Move on to next net {9}.\n'
                                'The reason is {5}, the exit code is {6}.\n'
                                'Total time this time for network {3}: {10}.\n'
                                'learning rate: [{7}] decay: [{8}]\n'
                            ).format(self.name, cnt, epoch, net, self.out_cls, message, code, self.learning_rate,
                                     self.decay, net + 1, str(timedelta(seconds=(time.time() - flag_2))).split('.')[0], fail_cnt)
                            mail(subject='%s failed' % self.name, pngs=pngs,
                                 dir_pic=directory,
                                 txts=[self.name + '.txt'], dir_txt=dir_txt, info=mail_body)
                            print(mail_body)
                            print ('Total training time for {0} network: {1}.\t').format(net + 1, str(
                                timedelta(seconds=(time.time() - flag_2))).split(
                                '.')[0])
                            epoch = self.epochs
                            # sys.exit('All the experiments have been failed, please try again!')
                            # mail(subject='%s failed' % self.name, pngs=pngs,
                            #      dir_pic=CURRENTDIR + "runs/%s/imgs/" % (self.name),
                            #      txts=self.name, dir_txt=CURRENTDIR, info=mail_body)
                    else:
                        # next iteration
                        self.__save_checkpoint({
                            'parameters': self.__parameters(prec1_ls, best_prec1_all_ls, epoch, net, model.state_dict(),
                                                            elbos, prec1_train, fail, message, code, seed)}, is_best, prec1,
                            net)
                        is_best_all = False
                        if epoch % 2 == 0:
                            print('\n')  # segment every two epochs
                        if epoch == self.epochs - 1:
                            # prec1_ls.append(prec1)
                            prec1_ls[net] = prec1
                            pngs = ['prec_net_{}_train_success.png'.format(net),
                                    'elbo_net_{}_train_success.png'.format(net),
                                    'lr_net_{}_success.png'.format(net)]
                            # self.plot(net, best_prec1_all_ls, name=pngs[0], caption='overall test accuracy')
                            # self.plot(net, prec1_ls, name=pngs[1], caption='single class accuracy')
                            self.plot(range(1, epoch + 2), prec1_train[net, 0:epoch + 1], name=pngs[0],
                                      caption='net %s train accuracy' % net, x_label='epoch')
                            self.plot(range(1, epoch + 2), elbos[net, 0:epoch + 1, 0], name=pngs[1],
                                      caption='net %s train elbo' % net,
                                      x_label='epoch', y_label='elbo')
                            self.__plot_lr(name=pngs[2])
                            mail_body = (
                            'Your training <<{0}>> has succeeded in net {1}, prec {2} after {3}({5}) times failures.\n'
                            'The model has been stored, the parameters have been recorded in outpout.\n'
                            'Total time this time for network {1}: {4}.\n'
                            ).format(self.name, net, prec1, cnt, str(timedelta(seconds=(time.time() - flag_2))).split('.')[0], fail_cnt)
                            print(mail_body)
                            mail(subject='%s succeed' % self.name, pngs=pngs, dir_pic=directory,
                            txts=[self.name + '.txt'], dir_txt=dir_txt, info=mail_body)
                        epoch += 1

                print ('Total time for network {3}: {2}.\t'
                       'Best single network accuracy among {0}: {1: .2f}').format(self.epochs, best_prec1,
                                                                                  str(timedelta(seconds=(
                                                                                      time.time() - flag_2))).split(
                                                                                      '.')[0], net)
                # print ('Total time for network {3}: {2:.1f} min.\t'
                #        'Best single network accuracy among {0}: {1: .2f}').format(self.epochs, best_prec1,
                #                                                                   (time.time() - flag_2) / 60, net)
                print ('Overall accuracy after {0} epochs: {1: .2f}, the best is {2: .2f}').format(self.epochs, prec1_all,
                                                                                                   best_prec1_all)
                print('-' * 50 + ' Network ' + str(net) + ' ' + "-" * 50)
                # self.plot(net, best_prec1_all_ls, name='overall', caption='overall test accuracy')
                # self.plot(net, prec1_ls, name='single', caption='single class accuracy')
                print('\n' * 2)  # segment different classes
            # print ('Total training time for {1} network: {0:.1f} min.\t').format((time.time() - flag_1) / 60,
            # the training has succeeded
            print ('Total training time for {0} network: {1}.\t').format(self.train_net, str(
                timedelta(seconds=(time.time() - flag_1))).split(
                '.')[0])

            pngs = ['overall.png', 'single.png']
            prec_ls, prec_all_ls = np.zeros(self.num_net), np.zeros(self.num_net),
            for net in range(self.num_net):
                x_testing, y_testing = shuffle(x_test_ord[net], y_test_ord[net], random_state=seed)
                x_val, y_val = shuffle(x_test[0: (net + 1) * self.test_pr_net],
                                       y_test[0: (net + 1) * self.test_pr_net], random_state=seed)
                testing = data_utils.TensorDataset(torch.FloatTensor(x_testing), torch.IntTensor(y_testing))
                testing_loader = data_utils.DataLoader(testing, batch_size=self.batch_size, shuffle=shuf)
                val = data_utils.TensorDataset(torch.FloatTensor(x_val), torch.IntTensor(y_val))
                val_loader = data_utils.DataLoader(val, batch_size=self.batch_size, shuffle=shuf)
                prec = self.__validate_net(testing_loader, model, net)
                prec_all = self.validate(val_loader, model, net)
                prec_ls[net] = prec
                prec_all_ls[net] = prec_all
            self.plot(range(self.out_cls, (net + 2) * self.out_cls, self.out_cls), prec_all_ls, name=pngs[0],
                      caption='overall test accuracy')
            self.plot(range(self.out_cls, (net + 2) * self.out_cls, self.out_cls), prec_ls, name=pngs[1],
                      caption='single class accuracy')
            self.__save_checkpoint({
                'parameters': self.__parameters(prec1_ls, best_prec1_all_ls, epoch, net, model.state_dict(),
                                                elbos, prec1_train, fail, message, code, seed)}, is_best=False, prec=0,
                net=0, filename='final_success_model')
            mail_body = (
                'Your training <<{0}>> has finished after {1} times failures.\n'
                'The model has been stored, the parameters have been recorded in outpout.\n'
                'Total time for network {2}: {3}.\n'
            ).format(self.name, fail_cnt, len(net_ls), str(timedelta(seconds=(time.time() - flag_1))).split('.')[0])
            print(mail_body)
            mail(subject='%s finish' % self.name, pngs=pngs, dir_pic=directory,
                 txts=[self.name + '.txt'], dir_txt=dir_txt, info=mail_body)
            # self.plot(epoch, prec1_train[net], name=pngs[2],
            #           caption='net {} train accuracy', x_label='epoch')
            # self.plot(epoch, elbos[net], name=pngs[3], caption='net {} train elbo',
            #           x_label='epoch', y_label='elbo')       #                                                                      self.train_net)

            # print(('=> Loaded Experiment parameters:\n'
            #    'elbo minus: [{0}]\t a: [{1}]\t b: [{2}]\t c: [{3}]\t'
            #    'batch: [{4}]\t lr: [{5}]\t epoch: [{6}]\t cls: [{7}]').format(self.minus, self.a, self.b, self.c, self.batch_size, self.learning_rate,
            #                                                                   self.epochs, self.train_net))

    def __load_parameters(self, pth):
        if os.path.isfile(pth):
            cp = torch.load(pth)
            try:
                self.data_pth = cp['data_pth']
                self.minus = cp['minus']
                self.factor = cp['f']
                self.a = cp['a']
                self.b = cp['b']
                self.c = cp['c']
                self.num_net = cp['num_net']
                self.out_cls = cp['out_cls']
                self.train_net = cp['train_net']
                self.train_pr_net = cp['train_pr_net']
                self.test_pr_net = cp['test_pr_net']
                self.num_features = cp['num_features']
                self.batch_size = cp['batch_size']
                self.epochs = cp['epochs']
                self.print_freq = cp['print_freq']
                self.parallel = cp['parallel']
                self.if_resume = cp['if_resume']
                self.hbnn = cp['hbnn']
                self.learning_rate = cp['learning_rate']
                self.decay = cp['decay']
                # these are parameters specific to training
                prec1_ls = cp['prec1_ls']
                best_prec1_all_ls = cp['best_prec1_all_ls']
                epoch = cp['epoch']
                net = cp['net']
                state_dict = cp['state_dict']
                elbos = cp['elbos']
                prec1_train = cp['prec_train']
                fail = cp['fail']
                message = cp['message']
                code = cp['code']
                seed = cp['seed']
            except KeyError:
                model = cp['hbnn']





    def train(self, dir_txt, seed=0, val_freq=10, start_net=0, shuf=True):
        """if_resume the training process from a certain model"""
        print('start over training from class 0')
        directory = CURRENTDIR + "/runs/%s/imgs/" % (self.name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        flag_1 = time.time()  # timing flag for whole trainning
        fail_cnt = 0
        # prec1_ls = list()  # store the accuracy for every single net
        # best_prec1_ls = list()  # store the best accuracy for every single net
        # best_prec1_all_ls = list()  # store the best accuracy for the whole net
        prec1_ls = np.zeros(self.train_net) # store the accuracy for every single net
        best_prec1_ls = np.zeros(self.train_net) # store the best accuracy for every single net
        best_prec1_all_ls = np.zeros(self.train_net)  # store the best accuracy for the whole net
        elbos = np.zeros((self.train_net, self.epochs, 4))
        prec1_train = np.zeros((self.train_net, self.epochs))
        model = Hbnn(self.out_cls)
        x_train_ord, y_train_ord, x_test_ord, y_test_ord, x_test, y_test = self.__load_data(self.data_pth)

        # if_resume the training
        for net in range(start_net, start_net + self.train_net):
            flag_2 = time.time()  # timing for a single net

            # prepare the training data
            if shuf:
                # data used to training
                x_training, y_training = shuffle(x_train_ord[net], y_train_ord[net], random_state=seed)
                # data used to test single class
                x_testing, y_testing = shuffle(x_test_ord[net], y_test_ord[net], random_state=seed)
                # data used to test the whole net
                x_val, y_val = shuffle(x_test[0: (net + 1) * self.test_pr_net],
                                       y_test[0: (net + 1) * self.test_pr_net], random_state=seed)
            else:
                x_training, y_training = x_train_ord[net], y_train_ord[net]
                x_testing, y_testing = x_test_ord[net], y_test_ord[net]
                x_val, y_val = x_test[0: (net + 1) * self.test_pr_net], y_test[0: (net + 1) * self.test_pr_net]
            # cut the data in minibatches and shuffle
            training = data_utils.TensorDataset(torch.FloatTensor(x_training), torch.IntTensor(y_training))
            training_loader = data_utils.DataLoader(training, batch_size=self.batch_size, shuffle=shuf)
            testing = data_utils.TensorDataset(torch.FloatTensor(x_testing), torch.IntTensor(y_testing))
            testing_loader = data_utils.DataLoader(testing, batch_size=self.batch_size, shuffle=shuf)
            val = data_utils.TensorDataset(torch.FloatTensor(x_val), torch.IntTensor(y_val))
            val_loader = data_utils.DataLoader(val, batch_size=self.batch_size, shuffle=shuf)

            # add a subnet for new class
            best_prec1 = 0  # best accuracy of a single net
            best_prec1_all = 0  # best accuracy for the whole network
            para_ls = iter(self.para_ls)
            self.learning_rate, self.decay = next(para_ls)
            # if self.lr_func == 0:
            #     self.learning_rate, self.decay = next(para_ls)
            # if self.lr_func == 1:
            #     self.learning_rate, self.decay = next(para_ls)
            # best_prec1_ls.append(best_prec1)
            # best_prec1_all_ls.append(best_prec1_all)

            # if not self.if_resume:
            model.addnet()

            # put the model on GPU, however, the multi-gpu is not conceivable at the moement
            if self.parallel:
                model_paral = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
            else:
                model.cuda()
            print('=> Number of trained parameters: {}'.format(
                sum([p.data.nelement() for p in model.params_net(net)])))
            print('=> Total model parameters increased to: {}'.format(
                sum([p.data.nelement() for p in model.parameters()])))

            # set the optimizer
            optimizer = torch.optim.RMSprop(model.params_net(net), lr=self.learning_rate, weight_decay=0.9)

            # segment of class
            print(' ' * 50 + ' Network ' + str(net) + ' ' + " " * 50)
            # for epoch in range(self.epochs):
            epoch = 0
            cnt = 0
            is_best_all = False
            prec1_all = 0
            best_prec1_all = 0
            while epoch < self.epochs:
                lr = self.__adjust_lr(optimizer, epoch)
                # train for one epoch
                e = self.__train(training_loader, model, optimizer, net, epoch)
                elbos[net, epoch] = np.asarray(e)
                print('Learning Rate: [{0:.4f}]').format(lr)
                # evaluate on validation set on a single class
                prec1 = self.__validate_net(testing_loader, model, net, epoch)
                prec1_train[net, epoch] = prec1
                # evaluate on the single class
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                if is_best:
                    best_prec1_ls[net] = best_prec1  # store the best prec

                # evaluate on the whole net
                if (epoch % val_freq == 0 and epoch != 0) or epoch == self.epochs - 1:
                    prec1_all = self.validate(val_loader, model, net, epoch)
                    is_best_all = prec1_all > best_prec1_all
                    best_prec1_all = max(prec1_all, best_prec1_all)
                    if is_best_all:
                        best_prec1_all_ls[net] = best_prec1_all

                # judge whether the training has failed
                fail, message, code = self.__judge(net, epoch, self.epochs, prec1_train, elbos)

                if fail:
                    filename = 'fail_{}_net_{}_epoch_{}'.format(fail_cnt, net, epoch)
                    self.__save_checkpoint({
                        'parameters': self.__parameters(prec1_ls, best_prec1_all_ls, epoch, net, model.state_dict(),
                                                        elbos, prec1_train, fail, message, code, seed)}, is_best, prec1,
                        net, filename)
                    # pngs = ['overall.png', 'single.png', 'prec_net_{}_train.png'.format(net), 'elbo_net_{}_train.png'.format(net)]
                    pngs = ['prec_net_{}_train_fail_{}.png'.format(net, fail_cnt),
                            'elbo_net_{}_train_fail_{}.png'.format(net, fail_cnt),
                            'lr_net_{}_fail_{}.png'.format(net, fail_cnt)]
                    self.plot(range(1, epoch + 2), prec1_train[net, 0:epoch + 1], name=pngs[0],
                              caption='net %s train accuracy' % net, x_label='epoch')
                    self.plot(range(1, epoch + 2), elbos[net, 0:epoch + 1, 0], name=pngs[1],
                              caption='net %s train elbo' % net,
                              x_label='epoch', y_label='elbo')
                    self.__plot_lr(name=pngs[2])
                    fail_cnt += 1
                    cnt += 1
                    try:
                        # reset the parameters and train again
                        lr, dc = self.learning_rate, self.decay
                        self.learning_rate, self.decay = next(para_ls)
                        model.reset(net)
                        model.cuda()
                        mail_body = (
                            'Your training <<{0}>> has failed {1}({12}) times in epoch {2}, net {3}, class batch {4}.\n'
                            'The model has been stored. The parameters have been adjusted and start to train this net again.\n'
                            'The reason is {5}, the exit code is {6}.\n'
                            'Total time this time for network {3}: {11}.\n'
                            'learning rate: [{7}] => [{8}]\t decay: [{9}] => [{10}]\n'
                        ).format(self.name, cnt, epoch, net, self.out_cls, message, code, lr, self.learning_rate,
                                 dc, self.decay, str(timedelta(seconds=(time.time() - flag_2))).split('.')[0], fail_cnt)
                        print(mail_body)
                        mail(subject='%s failed' % self.name, pngs=pngs,
                             dir_pic=directory,
                             txts=[self.name + '.txt'], dir_txt=dir_txt, info=mail_body)
                        epoch = 0
                        flag_2 = time.time()  # timing for a single net
                    except StopIteration:
                        mail_body = (
                            'Your training <<{0}>> has failed {1}({11}) times in epoch {2}, net {3}, class batch {4}.\n'
                            'The model has been stored. The parameters list has been exhausted and training has stopped.\n'
                            'Move on to next net {9}.\n'
                            'The reason is {5}, the exit code is {6}.\n'
                            'Total time this time for network {3}: {10}.\n'
                            'learning rate: [{7}] decay: [{8}]\n'
                        ).format(self.name, cnt, epoch, net, self.out_cls, message, code, self.learning_rate,
                                 self.decay, net + 1, str(timedelta(seconds=(time.time() - flag_2))).split('.')[0], fail_cnt)
                        mail(subject='%s failed' % self.name, pngs=pngs,
                             dir_pic=directory,
                             txts=[self.name + '.txt'], dir_txt=dir_txt, info=mail_body)
                        print(mail_body)
                        epoch = self.epochs
                        # sys.exit('All the experiments have been failed, please try again!')
                        # mail(subject='%s failed' % self.name, pngs=pngs,
                        #      dir_pic=CURRENTDIR + "runs/%s/imgs/" % (self.name),
                        #      txts=self.name, dir_txt=CURRENTDIR, info=mail_body)
                else:
                    # next iteration
                    self.__save_checkpoint({
                        'parameters': self.__parameters(prec1_ls, best_prec1_all_ls, epoch, net, model.state_dict(),
                                                        elbos, prec1_train, fail, message, code, seed)}, is_best, prec1,
                        net)
                    is_best_all = False
                    if epoch % 2 == 0:
                        print('\n')  # segment every two epochs
                    if epoch == self.epochs - 1:
                        # prec1_ls.append(prec1)
                        prec1_ls[net] = prec1
                        pngs = ['prec_net_{}_train_success.png'.format(net),
                                'elbo_net_{}_train_success.png'.format(net),
                                'lr_net_{}_success.png'.format(net)]
                        # self.plot(net, best_prec1_all_ls, name=pngs[0], caption='overall test accuracy')
                        # self.plot(net, prec1_ls, name=pngs[1], caption='single class accuracy')
                        self.plot(range(1, epoch + 2), prec1_train[net, 0:epoch + 1], name=pngs[0],
                                  caption='net %s train accuracy' % net, x_label='epoch')
                        self.plot(range(1, epoch + 2), elbos[net, 0:epoch + 1, 0], name=pngs[1],
                                  caption='net %s train elbo' % net,
                                  x_label='epoch', y_label='elbo')
                        self.__plot_lr(name=pngs[2])
                        mail_body = (
                        'Your training <<{0}>> has succeeded in net {1}, prec {2} after {3}({5}) times failures.\n'
                        'The model has been stored, the parameters have been recorded in outpout.\n'
                        'Total time this time for network {1}: {4}.\n'
                        ).format(self.name, net, prec1, cnt, str(timedelta(seconds=(time.time() - flag_2))).split('.')[0], fail_cnt)
                        print(mail_body)
                        mail(subject='%s succeed' % self.name, pngs=pngs, dir_pic=directory,
                        txts=[self.name + '.txt'], dir_txt=dir_txt, info=mail_body)
                    epoch += 1

            print ('Total time for network {3}: {2}.\t'
                   'Best single network accuracy among {0}: {1: .2f}').format(self.epochs, best_prec1,
                                                                              str(timedelta(seconds=(
                                                                                  time.time() - flag_2))).split(
                                                                                  '.')[0], net)
            # print ('Total time for network {3}: {2:.1f} min.\t'
            #        'Best single network accuracy among {0}: {1: .2f}').format(self.epochs, best_prec1,
            #                                                                   (time.time() - flag_2) / 60, net)
            print ('Overall accuracy after {0} epochs: {1: .2f}, the best is {2: .2f}').format(self.epochs, prec1_all,
                                                                                               best_prec1_all)
            print('-' * 50 + ' Network ' + str(net) + ' ' + "-" * 50)
            # self.plot(net, best_prec1_all_ls, name='overall', caption='overall test accuracy')
            # self.plot(net, prec1_ls, name='single', caption='single class accuracy')
            print('\n' * 2)  # segment different classes
        # print ('Total training time for {1} network: {0:.1f} min.\t').format((time.time() - flag_1) / 60,
        # the training has succeeded
        print ('Total training time for {0} network: {1}.\t').format(self.train_net, str(
            timedelta(seconds=(time.time() - flag_1))).split(
            '.')[0])

        pngs = ['overall.png', 'single.png']
        fail, message, code = False, 'success', 0
        self.plot(range(self.out_cls, (net + 2) * self.out_cls, self.out_cls), best_prec1_all_ls, name=pngs[0],
                  caption='overall test accuracy')
        self.plot(range(self.out_cls, (net + 2) * self.out_cls, self.out_cls), prec1_ls, name=pngs[1],
                  caption='single class accuracy')
        self.__save_checkpoint({
            'parameters': self.__parameters(prec1_ls, best_prec1_all_ls, epoch, net, model.state_dict(),
                                            elbos, prec1_train, fail, message, code, seed)}, is_best, prec1,
            net, filename='final_success_model')
        mail_body = (
            'Your training <<{0}>> has finished after {1} times failures.\n'
            'The model has been stored, the parameters have been recorded in outpout.\n'
            'Total time: {2}.\n'
        ).format(self.name, fail_cnt, str(timedelta(seconds=(time.time() - flag_1))).split('.')[0])
        print(mail_body)
        mail(subject='%s finish' % self.name, pngs=pngs, dir_pic=directory,
             txts=[self.name + '.txt'], dir_txt=dir_txt, info=mail_body)
        # self.plot(epoch, prec1_train[net], name=pngs[2],
        #           caption='net {} train accuracy', x_label='epoch')
        # self.plot(epoch, elbos[net], name=pngs[3], caption='net {} train elbo',
        #           x_label='epoch', y_label='elbo')       #                                                                      self.train_net)

    def plot(self, x, precls, name, caption, y_label='accuracy', x_label='class'):
        """draw the pic and save"""
        directory = CURRENTDIR + "/runs/%s/imgs/" % (self.name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # plt.plot(range(1 * self.out_cls, (net + 2) * self.out_cls, self.out_cls), precls, '.r--')
        plt.plot(x, precls, '.r--')
        ax = plt.gca()
        ax.yaxis.grid(True, linestyle='--')
        ax.xaxis.grid(True, linestyle='--')
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.suptitle(caption)
        # img = dir_pic + str(net) + '.png'
        img = directory + name
        plt.savefig(img, bbox_inches='tight')
        plt.close()
        print('picture saved @ %s' % img)

    def __resume(self):
        """if_resume the training process from a certain model"""
        x_train_ord, y_train_ord, x_test_ord, y_test_ord, x_test, y_test = self.__load_data(self.data_pth)
        # initialize the model
        model = self.hbnn
        # if there is saved model, load it.
        if self.if_resume:
            test = data_utils.TensorDataset(torch.FloatTensor(x_test), torch.IntTensor(y_test))
            val_loader = data_utils.DataLoader(test, batch_size=self.batch_size, shuffle=shuf)
            stop_epoch, best_prec_ls, start_cls, model = self.load_model(model=model, resume=self.if_resume,
                                                                         val_loader=val_loader)
        else:
            start_cls = 0

        # if_resume the training
        for cls in range(start_cls, self.train_net):
            # shuffle the training data
            if shuf:
                x_training, y_training = shuffle(x_train_ord[cls], y_train_ord[cls], random_state=0)
            else:
                x_training = x_train_ord[cls]
                y_training = y_train_ord[cls]

            # cut the data in minibatches and shuffle
            training = data_utils.TensorDataset(torch.FloatTensor(x_training), torch.IntTensor(y_training))
            training_loader = data_utils.DataLoader(training, batch_size=self.batch_size, shuffle=shuf)
            # add new net
            if self.if_resume and cls == start_cls:
                best_prec1_ls = best_prec_ls
                best_prec1 = best_prec_ls[start_cls]
                start_epoch = stop_epoch
                print('if_resume training from class {0}, epoch {1}').format(start_cls, start_epoch)
                print('=> Number of trained parameters: {}'.format(
                    sum([p.data.nelement() for p in model.params_net(cls)])))
                print('=> Number of model parameters: {}'.format(
                    sum([p.data.nelement() for p in model.parameters()])))
                # print('=> Number of model parameters by myself: {}'.format(
                #     sum([p.data.nelement() for p in model.params()])))
            else:
                best_prec1 = 0
                best_prec1_ls.append(best_prec1)
                start_epoch = 0
                model.addnet()
                model = model.cuda()
                print('if_resume training class {0}, epoch {1}').format(cls, start_epoch)
                print('=> Number of trained parameters: {}'.format(
                    sum([p.data.nelement() for p in model.params_net(cls)])))
                print('=> Number of model parameters: {}'.format(
                    sum([p.data.nelement() for p in model.parameters()])))
                # print('=> Number of model parameters by myself: {}'.format(
                #     sum([p.data.nelement() for p in model.params()])))

            # set the optimizer
            optimizer = torch.optim.RMSprop(model.params_net(cls), lr=self.learning_rate, weight_decay=0.9)

            # segment of class
            print(' ' * 50 + ' Class ' + str(cls) + ' ' + " " * 50)
            for epoch in range(start_epoch, self.epochs):

                self.__adjust_lr(optimizer, epoch)
                # train for one epoch
                self.__train(training_loader, model, optimizer, cls, epoch)
                # evaluate on validation set on a single class
                prec1 = self.__validate_net(val_loader, model, cls, epoch)

                # remember best prec@1 and save checkpoint
                is_best = prec1 > best_prec1
                if is_best:
                    best_prec1_ls[cls] = best_prec1  # store the best prec
                best_prec1 = max(prec1, best_prec1)
                self.__save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_prec1_ls': best_prec1_ls,
                    'cls': cls
                }, is_best, prec1, cls)
            print ('Best accuracy among {0}: {1: .2f}').format(self.epochs, best_prec1)
            print('\n')  # segment different classes

    def __train(self, training_loader, model, optimizer, net, epoch=0):
        batch_time = AverageMeter()
        fwd_time = AverageMeter()
        losses = AverageMeter()
        data_term = AverageMeter()
        entro = AverageMeter()
        cross = AverageMeter()
        top1 = AverageMeter()
        elbo = list()
        # switch to train mode
        model.train()

        # if_resume the timing
        for i, (input, target) in enumerate(training_loader):
            flag_1 = time.time()
            target = target.cuda(async=True)
            target = target - self.out_cls * net
            input = input.cuda()
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # get the elbo
            loss, d, e, c = model.elbo(input_var, target_var, minus=self.minus, factor=self.factor, a=self.a, b=self.b,
                                       c=self.c,
                                       train_net=net, batchs=len(training_loader))
            flag_2 = time.time()
            output = model.forward_single_net(input_var, net)
            fwd_time.update(time.time() - flag_2)
            prec1 = self.__accuracy_net(output.data, target, num=net, topk=(1,))[0]
            # update the data
            losses.update(loss.data[0], input.size(0))
            data_term.update(d.data[0], input.size(0))
            entro.update(e.data[0], input.size(0))
            cross.update(c.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            # measure elapsed time
            batch_time.update(time.time() - flag_1)
            # end = time.time()
            # print the result
            if i == 0:
                print('-' * 50 + ' Training ' + '-' * 50)
            if i % self.print_freq == 0 or i == len(training_loader) - 1:
                print(
                    'Class: [{3}/{4}][{5}/{6}]\t'
                    'Epoch: [{0}][{1}/{2}]\t'
                    'Time: {batch_time.val:.1f} ({batch_time.avg:.1f}|{fwd_time.avg:.1f})\t'
                    'Loss: {loss.val:.2f} ({loss.avg:.1f})\t'
                    'Prec@1: {top1.val:.2f} ({top1.avg:.2f})'.format(
                        epoch + 1, i + 1, len(training_loader), net + 1, self.train_net, (net + 1) * self.out_cls,
                                                                             self.train_net * self.out_cls,
                        batch_time=batch_time, fwd_time=fwd_time,
                        loss=losses, top1=top1))
                print ('data: {0:.1f}({3:.1f})\t  cross: {2:.1f}({5:.1f})\t entro: {1:.1f}({4:.1f})\t').format(
                    data_term.val, entro.val,
                    cross.val, data_term.avg,
                    entro.avg, cross.avg)
        elbo.append([losses.avg, data_term.avg, entro.avg, cross.avg])
        return elbo

    def __validate_net(self, val_loader, model, net, epoch=0):
        """Perform validation on the validation set"""
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        # switch to evaluate mode
        model.eval()

        num_batches = len(val_loader)  # the number of batches
        for i, (input, target) in enumerate(val_loader):
            start = time.time()
            target = target.cuda(async=True)
            target = target - net * self.out_cls
            input = input.cuda()
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)
            # compute output
            output = model.forward_single_net(input_var, net)
            # get the loss
            # loss, d, e, c = model.elbo(input_var, target_var, train_net=net, factor=self.factor, batchs=num_batches)
            # measure accuracy and record loss
            prec1 = self.__accuracy_net(output.data, target, net, topk=(1,))[0]
            # losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - start)
            end = time.time()
            # print the result
            if i == 0:
                print('-' * 50 + ' Testing ' + '-' * 50)
            if i % self.print_freq == 0 or i == len(val_loader) - 1:
                print(
                    'Class: [{3}/{4}][{5}/{6}]\t'
                    'Epoch: [{0}][{1}/{2}]\t'
                    'Time: {batch_time.val:.1f} ({batch_time.avg:.1f})\t'
                    # 'Loss: {loss.val:.2f} ({loss.avg:.1f})\t'
                    'Prec@1: {top1.val:.2f} ({top1.avg:.2f})'.format(
                        epoch + 1, i + 1, len(val_loader), net + 1, self.train_net, (net + 1) * self.out_cls,
                                                                        self.train_net * self.out_cls,
                        batch_time=batch_time,
                        loss=losses, top1=top1))
        print(' * Prec@1 {top1.avg:.2f}'.format(top1=top1))

        return top1.avg

    def __accuracy_cls(self, output, target, cls, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)
        target = target.eq(cls).long()
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = -1 * pred.eq(target.view(1, -1).expand_as(pred)) + 1
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def __accuracy_net(self, output, target, num, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def __adjust_lr(self, optimizer, epoch):
        """adjust learning rate according to epoch"""
        # a = 7e-2
        a = self.learning_rate
        # self.turn_pt_1 = 70
        # self.turn_pt_2 = 120
        # lr = a * (b + (epoch + 1)) ** (- 0.55)
        # lr = a * (b + (epoch + 1)) ** (-0.55)
        if self.lr_func == 0:
            b = - 0.5
            lr = a * (b + (epoch + self.start_epoch)) ** self.decay
        if self.lr_func == 1:
            b = self.decay
            # turn_point_1 = 100
            turn_point_1 = self.turn_pt_1
            turn_point_2 = self.turn_pt_2
            lr = a * b ** (epoch // turn_point_1) * b ** (epoch // turn_point_2)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def __plot_lr(self, name, plot_range=200):
        a = self.learning_rate
        x = np.arange(plot_range)
        if self.lr_func == 0:
            b = - 0.5
            y = a * (b + (x + self.start_epoch)) ** self.decay
        if self.lr_func == 1:
            b = self.decay
            # turn_point_1 = 100
            turn_point_1 = self.turn_pt_1
            turn_point_2 = self.turn_pt_2
            y = a * b ** (x // turn_point_1) * b ** (x // turn_point_2)
        self.plot(x, y, name, 'learning rate', y_label='lr', x_label='epoch')

    def __save_checkpoint(self, state, is_best, prec, net, filename='checkpoint'):
        """Saves checkpoint to disk"""
        # save current epoch
        directory = CURRENTDIR + "/runs/%s/" % (self.name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_name = directory + filename + '.pth.tar'
        torch.save(state, file_name)

        # save the best model
        if is_best:
            pth = directory + 'model_best' + '_net_' + str(net) + '.pth.tar'
            shutil.copyfile(file_name, pth)
            print('net [{0}]\t prec@[{1: .2f}]\t checkpoint saved at :{2}').format(net, prec, pth)

    def load_model(self, x_test, y_test, model=Hbnn(), resume=False, val_loader=False):
        """optionally if_resume from a checkpoint and test it"""
        if resume:
            if os.path.isfile(resume):
                print("=> loading checkpoint '{}'".format(resume))
                checkpoint = torch.load(resume)
                start_epoch = checkpoint['epoch']
                best_prec1_ls = checkpoint['best_prec1_ls']
                cls = checkpoint['cls']
                # parameters = checkpoint['parameters']
                # self.__reset_params()
                for i in range(cls + 1):
                    model.addnet()
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(resume, checkpoint['epoch']))
                model.cuda()
                # print(('=> Loaded Experiment parameters:\n'
                #    'elbo minus: [{0}]\t a: [{1}]\t b: [{2}]\t c: [{3}]\t'
                #    'batch: [{4}]\t lr: [{5}]\t epoch: [{6}]\t cls: [{7}]').format(self.minus, self.a, self.b, self.c, self.batch_size, self.learning_rate,
                #                                                                   self.epochs, self.train_net))

                # test the resumed model
                if val_loader:
                    print(' ' * 30 + ' Testing loaded model from class {0} to class {1} ' + ' ' * 40).format(0, cls)
                    print('=> Number of loaded model parameters by default: {}'.format(
                        sum([p.data.nelement() for p in model.parameters()])))
                    print('=> Number of loaded model parameters by myself: {}'.format(
                        sum([p.data.nelement() for p in model.params()])))
                    x_val, y_val = shuffle(x_test[0: (cls + 1) * self.test_pr_net],
                                           y_test[0: (cls + 1) * self.test_pr_net], random_state=0)
                    val = data_utils.TensorDataset(torch.FloatTensor(x_val), torch.IntTensor(y_val))
                    val_loader = data_utils.DataLoader(val, batch_size=self.batch_size, shuffle=shuf)
                    self.validate(val_loader=val_loader, net=cls, model=model)
                    # for i in range(cls + 1):
                    #     prec1 = self.__validate_net(val_loader, model, cls=i, epoch=0)
                    #     print('\n')

                return start_epoch, best_prec1_ls, cls, model
            else:
                print("=> no checkpoint found at '{}'".format(resume))


                # def test_resume(self, val_loader, model, start_cls, epoch=0):
                #     """Testing"""
                #     print('=> Number of model parameters by default: {}'.format(
                #         sum([p.data.nelement() for p in model.parameters()])))
                #     print('=> Number of model parameters by myself: {}'.format(
                #         sum([p.data.nelement() for p in model.params()])))
                #     for cls in range(start_cls + 1):
                #         prec1 = self.__validate_net(val_loader, model, epoch, cls)
                #
                # def test_parameters(self):
                #     model = Hbnn()
                #     cudnn.benchmark = True
                #     for i in range(self.num_net):
                #         model.addnet()
                #         model = model.cuda()
                #         print('=> Number of model parameters: {}'.format(
                #             sum([p.data.nelement() for p in model.parameters()])))
                #         print('=> Number of model parameters: {}'.format(
                #             sum([p.data.nelement() for p in model.params()])))

    def validate(self, val_loader, model, net, epoch=0):
        """Perform validation on the validation set"""
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        # switch to evaluate mode
        model.eval()

        num_batches = len(val_loader)  # the number of batches
        for i, (input, target) in enumerate(val_loader):
            start = time.time()
            target = target.cuda(async=True)
            input = input.cuda()
            input_var = torch.autograd.Variable(input, volatile=True)
            # target_var = torch.autograd.Variable(target - net * out_cls, volatile=True)
            # compute output
            output = model.forward(input_var, net)
            # measure accuracy and record loss
            prec1 = self.accuracy(output=output.data, target=target, topk=(1,))[0]
            # get the loss
            # loss, d, e, c = model.elbo(input_var, target_var, train_net=net, factor=self.factor, batchs=num_batches)
            # losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - start)
            # print the result
            if i == 0:
                print('-' * 50 + ' Testing ' + '-' * 50)
            if i % self.print_freq == 0 or i == len(val_loader) - 1:
                print(
                    'Total Classes: [{3}][{4}]\t'
                    'Epoch: [{0}][{1}/{2}]\t'
                    'Time: {batch_time.val:.1f} ({batch_time.avg:.1f})\t'
                    # 'Loss: {loss.val:.2f} ({loss.avg:.1f})\t'
                    'Overall Prec@1: {top1.val:.2f} ({top1.avg:.2f})'.format(
                        epoch + 1, i + 1, len(val_loader), net + 1, (net + 1) * self.out_cls, batch_time=batch_time,
                        loss=losses, top1=top1))
        print(' * Overall Prec@1 {top1.avg:.2f}'.format(top1=top1))

        return top1.avg

    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k for the whole network"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def __judge(self, net, epoch, epochs, prec_train, elbos, stag_epochs=20, epoch_lb=20, epoch_trap=40,
                prec_trap=5, threshold=70):
        """check the status of training, quit if fails with notification

        Conditions that fails
        1. stagnation, no training at all
        2. trap, trapped in a local extreme
        3. divergence, the elbo diverges
        """
        precs = prec_train[net, 0: epoch + 1]
        prec = prec_train[net, epoch]
        prec_lb = (1 / self.out_cls + 0.02) * 100
        if epoch >= epoch_lb - 1:
            if prec < prec_lb:
                fail = True
                message = 'no training at all in {} epochs'.format(epoch_lb)
                code = 1
                return fail, message, code
        if epoch >= epoch_trap:
            prec_1 = prec_train[net, epoch - epoch_trap]
            prec_2 = prec_train[net, epoch]
            if prec_2 - prec_1 < prec_trap and prec_2 < threshold:
                fail = True
                message = 'trapped in a local extreme {} for {} epochs'.format(prec_2, epoch_trap)
                code = 2
                return fail, message, code
        elbo = elbos[net, epoch][0]
        if math.isnan(elbo):
            fail = True
            message = 'the elbo diverges at epoch {}'.format(epoch)
            code = 3
            return fail, message, code

        if DEBUG:
            return True, 'test', -1
        return False, 'normal training', 0

