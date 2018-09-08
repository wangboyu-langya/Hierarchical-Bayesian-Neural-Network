import math
import torch
import torch.nn as nn
from torch import log
from torch.autograd import Variable
from torch.nn import ModuleList
from torch.nn import ParameterList
from torch.nn.parameter import Parameter
import numpy as np

CONS = math.log(2 * math.pi)


class Lrpmlinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Lrpmlinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.mu = Parameter(torch.Tensor(in_features, out_features)).cuda()  # this is the mean of weight w
        # self.mu = Parameter(torch.Tensor(in_features, out_features)).cuda()  # this is the mean of weight w
        self.mu = Parameter(torch.ones(in_features, out_features))  # this is the standard deviation
        self.sigma = Parameter(torch.ones(in_features, out_features))  # this is the standard deviation
        # self.mu = Parameter(torch.randn(in_features, out_features)).cuda()  # this is the mean of weight w
        # self.sigma = Parameter(torch.randn(in_features, out_features)).cuda()  # this is the standard deviation
        # self.nelements = Variable(torch.LongTensor(self.mu.nelement())).cuda()  # the number of node weights
        self.nelements = self.mu.nelement()  # the number of node weights

    def sigma_repara(self):
        return log(1 + torch.exp(self.sigma))

    def reset_parameters(self):
        '''reset the parameters as standard Gaussian distribution'''
        self.mu.data.fill_(0)
        self.sigma.data.fill_(1)

    def forward(self, input):
        """the forward propagation is different from normal neural networks, since we are going to sample the weights,
        the dimension of weights are usually much larger than the output, so here we sample the output instead of the weights

        See 2015 - Kingma, Salimans, Welling - Variational Dropout and the Local Reparameterization Trick for details,
        https://arxiv.org/abs/1506.02557.
        """
        gamma = input.matmul(
            self.mu)  # be aware matmul has to be applied to matrix only, for tensors it should be mm.
        delta = input.pow(2).matmul(self.sigma_repara().pow(2))
        zeta = Variable(torch.randn(delta.data.shape), requires_grad=False)
        if torch.cuda.is_available():
            zeta = zeta.cuda()
        output = gamma + delta.sqrt() * zeta
        # output = output  # all the variables here are temporary, so there is no need to differentiate
        return output


class Net(nn.Module):
    def __init__(self, out_cls=10):
        super(Net, self).__init__()
        self.out_cls = out_cls
        self.fc1 = Lrpmlinear(456, 400)
        # self.fc2 = Lrpmlinear(400, 400)
        self.fc3 = Lrpmlinear(400, out_cls)
        self.relu = nn.ReLU(inplace=True)
        self.sm = nn.Softmax()
        self.nelements = self.fc1.nelements + self.fc3.nelements

    def reset(self):
        self.fc1 = Lrpmlinear(456, 400)
        # self.fc2 = Lrpmlinear(400, 400)
        self.fc3 = Lrpmlinear(400, self.out_cls)

    def params(self):
        """self implemented, equivalent to model.parameters """
        parameters = list()
        [parameters.append(i) for i in self.fc1.parameters()]
        [parameters.append(i) for i in self.fc3.parameters()]
        return parameters

    def mu_square(self):
        mu_square = self.fc1.mu.pow(2).sum() \
                    + self.fc3.mu.pow(2).sum()
        # + self.fc2.mu.pow(2).sum()\
        return mu_square

    def sigma_square(self):
        sigma_square = self.fc1.sigma_repara().sum() + self.fc3.sigma_repara().sum()
            # + self.fc3.sigma_repara().sum() # replace square with reparametrization
        return sigma_square

    def forward(self, x):
        """Be aware the input has to be a Variable, the dimension should be n x 456"""
        out = self.relu(self.fc1(x))
        # out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.sm(out)
        return out


class Hbnn(nn.Module):
    def __init__(self, out_cls=10):
        # these are two useless prior
        super(Hbnn, self).__init__()
        self.prior_v = 100
        self.prior_tau_0_reciprocal = 1000
        self.num_net = 0
        self.out_cls = out_cls
        self.w0 = Net(out_cls)  # this is the network of w0
        self.hbnn = ModuleList()  # this is the network of all the classes
        self.mu_gamma_g = ParameterList()
        self.sigma_gamma_g = ParameterList()
        self.mu_gamma = Parameter(torch.ones(1))
        self.sigma_gamma = Parameter(torch.ones(1))

        if torch.cuda.is_available():
            self.w0 = self.w0.cuda()

    def params(self):
        """self implemented method to check the number of all the parameters"""
        parameters = list()
        [parameters.append(i) for i in self.w0.params()]
        [parameters.append(i) for i in self.mu_gamma_g]
        [parameters.append(i) for i in self.sigma_gamma_g]
        for j in self.hbnn:
            [parameters.append(i) for i in j.params()]

        return parameters

    def params_net(self, num):
        """self implemented method to check the number of parameters of single class"""
        parameters = list()
        parameters.append(self.mu_gamma_g[num])
        parameters.append(self.sigma_gamma_g[num])
        [parameters.append(i) for i in self.hbnn[num].params()]
        [parameters.append(i) for i in self.w0.params()]

        return parameters


    def resume_cuda(self):
        """make every parameter on cuda after if_resume"""
        for p in self.mu_gamma_g:
            p = p.cuda()
        for p in self.sigma_gamma_g:
            p = p.cuda()

    def addnet(self):
        self.hbnn.append(Net(self.out_cls).cuda())
        self.num_net += 1
        self.mu_gamma = Parameter(torch.ones(1))
        self.sigma_gamma = Parameter(torch.ones(1))
        self.mu_gamma_g.append(self.mu_gamma)
        self.sigma_gamma_g.append(self.sigma_gamma)

    def forward_single_net(self, x, num, mc_times=100):
        """forward through the current net"""
        net = self.hbnn[num]
        output = 0
        for i in range(mc_times):
            out = net.forward(x)
            output += out
        return output / mc_times

    def forward(self, x, num, mc_times=100):
        """forward through the whole net"""
        # assert (self.num_net - 1 == num), 'the number of testing classes is {0}, while the net has {1}'.format(num, self.num_net)
        output = self.forward_single_net(x, num=0, mc_times=mc_times)
        # if num != 0:
        #     output = output[:, 0].unsqueeze(1)
        # output = self.forward_single_net(x, num=0, mc_times=mc_times).t()[0, :]
        for n in range(1, num + 1):
            # _ = self.forward_single_net(x, num=num, mc_times=mc_times).t()[0, :]
            _ = self.forward_single_net(x, num=n, mc_times=mc_times)
            # _ = _.view(1, -1)
            # _ = _[:, 0].unsqueeze(1)
            # if num == 1:
            #     output = torch.stack((output, _))
            # else:
            #     _ = _.unsqueeze(0)
            #     output = torch.cat((output, _), dim=0)
            output = torch.cat((output, _), dim=1)
        return output


    def elbo(self, x, y, batchs, train_net, minus=True, factor=2, a=10e1, b=10e1, c=10e2, mc_times=100, if_debug=False):
        """this calculates the elbo of current network, which is unrelated to the data"""
        d = self._data_term(x, y, train_net=train_net, factor=factor, mc_times=mc_times) * a
        # d.backward()
        e = self._ent_term(num=train_net) * b
        # e.backward()
        c = self._cross_ent_term(num=train_net) * c
        if_debug = False
        if if_debug:
            print ('data :{0}\t entro :{1}\t cross :{2}\t').format(d.data.cpu().numpy()[0], e.data.cpu().numpy()[0],
                                                                   c.data.cpu().numpy()[0])
            # print('d is: {0}').format(d.data.cpu().numpy())
            # print('e is: {0}').format(e.data.cpu().numpy())
            # print('c is: {0}').format(c.data.cpu().numpy())
        # c.backward()
        elbo = d + 1.0 / batchs * (c + e)
        # elbo = elbo * 100
        if minus:
            return -elbo, -d, -e, -c
        else:
            return elbo, d, e, c
        # return d + 1.0 / batchs * (e)
        # return d + 1.0 / batchs * (c)
        # return c

    def _data_term(self, x, y, train_net, factor=1, mc_times=100, if_debug=False):
        # Fixed: the inner_product is not right here at the moment, have to process the target
        inner_product = 0
        num_weights = self.w0.nelements
        net = self.hbnn[train_net]
        # process the target data
        col = y.cpu().data.numpy()
        row = np.arange(len(col))
        target = np.zeros((y.size(0), self.out_cls))
        target[row, col] = 1

        # target = np.zeros((y.size(0), self.out_cls))
        # target[:, col] = 1
        target = Variable(torch.Tensor(target)).cuda()

        for i in range(mc_times):
            output = net.forward(x)
            if if_debug:
                print('y is {0}'.format(y.cpu().data.numpy()))
            batch_size = y.size(0)
            inner_product += (output * target).sum()
            # len = y.data.shape[0]
            # index = -1 * y.eq(self.num_net - 1).long() + 1
            # index = index.data.cpu().numpy().reshape(len, )
            # y_reshape = np.zeros((len, 2))
            # update_values = np.ones((len))
            # y_reshape[np.arange(0, len), index] = update_values
            # y_reshape[:, 0] = y_reshape[:, 0] * factor
            # target = Variable(torch.Tensor(y_reshape)).cuda()
            # inner_product += (output * target).sum()
            # inner_product += output.gather(1, y.view(-1, 1)).sum()
            # correct = pred.eq(y.view(-1, 1).expand_as(pred)).float()
            # if if_debug:
            #     print('output is {0}'.format(output.cpu().data.numpy()))
            #     print('pred is {0}'.format(pred.t().cpu().data.numpy()))
            #     print('prob is {0}'.format(prob.t().cpu().data.numpy()))
            #     print('y is {0}'.format(y.cpu().data.numpy()))
            #     print('correct is {0}'.format(correct.t().cpu().data.numpy()))
            #     print('inner product is {0}'.format((prob * correct).sum()))
            # inner_product += (prob * correct).sum()

        # return inner_product / mc_times / num_weights
        return inner_product / mc_times

    def _cross_ent_term(self, num, batch_size=64, feature_size=456, mc_times=1000, if_debug=False):
        """Here er've got monte carlo for gamma_g"""
        num_weights = self.w0.nelements
        _1 = - 0.5 * 1 / self.prior_tau_0_reciprocal * (self.w0.mu_square().sum() + self.w0.sigma_square().sum()) \
             - 0.5 * num_weights * math.log(self.prior_tau_0_reciprocal) - 0.5 * num_weights * math.log(2 * math.pi)
        _1 = _1 / num_weights
        _2 = 0
        _3 = 0
        for g in range(num + 1):
            net = self.hbnn[g]
            epsilon_mc = self.mu_gamma_g[g] + log(1 + torch.exp(self.sigma_gamma_g[g])) * Variable(
                torch.randn(mc_times).cuda(), requires_grad=False)
            epsilon_mc = log(epsilon_mc ** 2).mean()
            _4 = - 0.5 * net.nelements * epsilon_mc
            _4 = _4 / num_weights
            # _4.backward()
            _5 = - 0.5 * (self.mu_gamma_g[g].pow(2) + log(1 + torch.exp(self.sigma_gamma_g[g])).pow(2))
            _5 = _5 / num_weights
            # _5.backward()
            # _6 = net.mu_square + net.sigma_square + self.w0.mu_square + self.w0.sigma_square - 0.5 * net.nelements * CONS
            _6 = net.mu_square() + net.sigma_square() + self.w0.mu_square() + self.w0.sigma_square()
            _6 = _6 / num_weights
            # _6.backward()
            _7 = - 2 * (torch.sum(net.fc1.mu * self.w0.fc1.mu) +
                        torch.sum(net.fc3.mu * self.w0.fc3.mu)) \
                # +  torch.sum(net.fc2.mu * self.w0.fc2.mu)
            # _7.backward()
            _7 = _7 / num_weights
            _9 = - 0.5 * net.nelements * CONS
            _9 = _9 / num_weights
            _8 = _5 * (_6 + _7) * num_weights
            # _9 = _4 + _8
            _2 = _2 + _4 + _8 + _9
            # _2 = _2 + (
            #     - 0.5 * net.nelements * torch.mean(log(epsilon_mc.pow(2)))
            #     - 0.5 * (self.mu_gamma_g[g].pow(2) + self.sigma_gamma_g[g].pow(2))
            #     * (
            #         net.mu_square() + net.sigma_square() + self.w0.mu_square() + self.w0.sigma_square() \
            #         - 0.5 * net.nelements * CONS - 2 * (torch.sum(net.fc1.mu * self.w0.fc1.mu) + \
            #                                             torch.sum(net.fc2.mu * self.w0.fc2.mu) + \
            #                                             torch.sum(net.fc3.mu * self.w0.fc3.mu))
            #     )
            # )
            _3 = _3 - 0.5 * CONS - 0.5 * math.log(self.prior_v) - 0.5 * 1.0 / self.prior_v * (
                self.mu_gamma_g[g].pow(2) + log((1 + torch.exp(self.sigma_gamma_g[g]).pow(2))))
            _3 = _3 / num_weights

        # if_debug = True
        if if_debug == True:
            print('_1 is {0}'.format(_1.cpu().data.numpy()))
            print('_2 is {0}'.format(_2.cpu().data.numpy()))
            print('_3 is {0}'.format(_3.cpu().data.numpy()))
            print('_4 is {0}'.format(_4.cpu().data.numpy()))
            print('_5 is {0}'.format(_5.cpu().data.numpy()))
            print('_6 is {0}'.format(_6.cpu().data.numpy()))
            print('_7 is {0}'.format(_7.cpu().data.numpy()))
            print('_8 is {0}'.format(_8.cpu().data.numpy()))
            # print('_9 is {0}'.format(_9.cpu().data.numpy()))
            print('_9 is {0}'.format(_9))
        return _1 + (_2 + _3) / (num + 1)
        # return (_1 + _2 + _3) * num_weights

    def _ent_term(self, num, if_debug=False):
        num_weights = self.w0.nelements
        _1 = 0.5 * num_weights * (CONS + 1) + log(self.w0.fc1.sigma_repara()).sum() + log(
            self.w0.fc3.sigma_repara()).sum() \
            # + log(self.w0.fc3.sigma_repara()).sum()
        _1 = _1 / num_weights
        _2 = 0
        _3 = 0
        for g in range(num + 1):
            net = self.hbnn[g]
            _2 = _2 + ((0.5 * net.nelements * (CONS + 1)) + log(net.fc1.sigma_repara()).sum() + log(
                net.fc3.sigma_repara()).sum()) / num_weights
            _3 = _3 + (0.5 * (CONS + 1)) + log(log(1 + torch.exp(self.sigma_gamma_g[g]))) / num_weights

        # if_debug = False
        if if_debug == True:
            print('_1 is {0}'.format(_1.cpu().data.numpy()))
            print('_2 is {0}'.format(_2.cpu().data.numpy()))
            print('_3 is {0}'.format(_3.cpu().data.numpy()))
        return _1 + (_2 + _3) / (num + 1)
        # return (_1 + _2 + _3) * num_weights

    def reset(self, net):
        network = self.hbnn[net]
        network.reset()
