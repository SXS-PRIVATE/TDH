# -*- coding: utf-8 -*-
# @Time    : 2023/10/26
# @Author  : SXS
# @Github  : https://github.com/SXS-PRIVATE
import numpy as np
import torch
from torch.optim import SGD
from torch.optim import Adam
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data import DataLoader

from torchcmh.dataset import triplet_data
from torchcmh.models import cnnf, MLP
from torchcmh.training.base import TrainBase
from torchcmh.utils import calc_neighbor


class TDH(TrainBase):
    """
    Yang et al. Pairwise relationship guided deep hashing for cross-modal retrieval.
    In Thirty-First AAAI Conference on Artificial Intelligence.2017
    """

    def __init__(self, data_name, img_dir, bit, visdom=True, batch_size=128, cuda=True, **kwargs):
        super(TDH, self).__init__("TDH", data_name, bit, batch_size, visdom, cuda)
        self.train_data, self.valid_data = triplet_data(data_name, img_dir, batch_size=batch_size, **kwargs)
        self.loss_store = ["inter_loss", "intra_loss", "quantization", "balance", "graph_regularization_loss",
                           "loss"]
        self.parameters = {'gamma': 100, 'eta': 50, 'beta': 1, 'alpha': 1}
        self.max_epoch = 500
        self.lr = {'img': 10 ** (-1.5), 'txt': 10 ** (-1.5)}
        self.lr_decay_freq = 1
        self.lr_decay = (1e-6 / 10 ** (-1.5)) ** (1 / self.max_epoch)
        self.num_train = len(self.train_data)
        self.img_model = cnnf.get_cnnf(bit)
        self.txt_model = MLP.MLP(self.train_data.get_tag_length(), bit, leakRelu=False)
        self.F_buffer = torch.randn(self.num_train, bit)
        self.G_buffer = torch.randn(self.num_train, bit)
        self.train_L = self.train_data.get_all_label()
        self.ones = torch.ones(self.batch_size, 1)
        self.ones_ = torch.ones(self.num_train - batch_size, 1)
        if cuda:
            self.img_model = self.img_model.cuda()
            self.txt_model = self.txt_model.cuda()
            self.train_L = self.train_L.cuda()
            self.F_buffer = self.F_buffer.cuda()
            self.G_buffer = self.G_buffer.cuda()
            self.ones = self.ones.cuda()
            self.ones_ = self.ones_.cuda()
        self.Sim = calc_neighbor(self.train_L, self.train_L)
        self.laplacian = torch.diag(torch.sum(self.Sim, dim=1)) - self.Sim
        self.B = torch.sign(self.F_buffer + self.G_buffer)
        optimizer_img = SGD(self.img_model.parameters(), lr=self.lr['img'])
        optimizer_txt = SGD(self.txt_model.parameters(), lr=self.lr['txt'])
        self.optimizers = [optimizer_img, optimizer_txt]
        self._init()

    def train(self, num_works=4):
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, drop_last=True, num_workers=num_works,
                                  shuffle=False, pin_memory=True)
        for epoch in range(self.max_epoch):
            self.img_model.train()
            self.txt_model.train()
            self.train_data.img_load()
            self.train_data.re_random_item()
            ones = torch.ones(self.num_train, 1).cuda()
            self.B = torch.sign(
                torch.matmul((self.F_buffer + self.G_buffer).t(),
                             torch.inverse(
                                 2 * ones + (self.parameters['beta'] / self.parameters['gamma']) * self.laplacian)
                             ).t()
            )
            for data in tqdm(train_loader):
                ind = data['index'].numpy()
                sample_L = data['label']  # type: torch.Tensor
                image = data['img']  # type: torch.Tensor
                pos_index = data['pos_index'].numpy()
                neg_index = data['neg_index'].numpy()
                pos_label = data['pos_label']  # type: torch.Tensor
                neg_label = data['neg_label']  # type: torch.Tensor
                pos_img = data['pos_img']
                neg_img = data['neg_img']
                if self.cuda:
                    image = image.cuda()
                    pos_img = pos_img.cuda()
                    neg_img = neg_img.cuda()
                    sample_L = sample_L.cuda()
                    pos_label = pos_label.cuda()
                    neg_label = neg_label.cuda()
                cur_f = self.img_model(image)
                cur_f_pos = self.img_model(pos_img)
                cur_f_neg = self.img_model(neg_img)
                self.F_buffer[ind, :] = cur_f.data
                self.F_buffer[pos_index, :] = cur_f_pos.data
                self.F_buffer[neg_index, :] = cur_f_neg.data

                F = Variable(self.F_buffer)
                G = Variable(self.G_buffer)

                inter_loss, intra_loss, quantization, balance, graph_regularization_loss = self.object_function(
                    cur_f, sample_L, G, F, ind,
                    pos_index, neg_index, pos_label, neg_label)
                loss = inter_loss + intra_loss + quantization + balance + graph_regularization_loss
                self.remark_loss(inter_loss, intra_loss, quantization, balance, graph_regularization_loss, loss)
                self.optimizers[0].zero_grad()
                loss.backward()
                self.optimizers[0].step()
            self.print_loss(epoch)
            self.plot_loss("img loss")
            self.reset_loss()

            self.train_data.txt_load()
            self.train_data.re_random_item()
            for data in tqdm(train_loader):
                ind = data['index'].numpy()
                sample_L = data['label']  # type: torch.Tensor
                text = data['txt']  # type: torch.Tensor
                pos_index = data['pos_index'].numpy()
                neg_index = data['neg_index'].numpy()
                pos_label = data['pos_label']  # type: torch.Tensor
                neg_label = data['neg_label']  # type: torch.Tensor
                pos_txt = data['pos_txt']
                neg_txt = data['neg_txt']
                if self.cuda:
                    text = text.cuda()
                    pos_txt = pos_txt.cuda()
                    neg_txt = neg_txt.cuda()
                    sample_L = sample_L.cuda()
                    pos_label = pos_label.cuda()
                    neg_label = neg_label.cuda()

                cur_g = self.txt_model(text)
                cur_g_pos = self.txt_model(pos_txt)
                cur_g_neg = self.txt_model(neg_txt)
                self.G_buffer[ind, :] = cur_g.data
                self.G_buffer[pos_index, :] = cur_g_pos.data
                self.G_buffer[neg_index, :] = cur_g_neg.data
                F = Variable(self.F_buffer)
                G = Variable(self.G_buffer)

                inter_loss, intra_loss, quantization, balance, graph_regularization_loss = self.object_function(
                    cur_g, sample_L, F, G, ind,
                    pos_index, neg_index, pos_label, neg_label)
                loss = inter_loss + intra_loss + quantization + balance #+ graph_regularization_loss
                self.remark_loss(inter_loss, intra_loss, quantization, balance, graph_regularization_loss, loss)
                self.optimizers[1].zero_grad()
                loss.backward()
                self.optimizers[1].step()

            self.print_loss(epoch)
            self.plot_loss("txt loss")
            self.reset_loss()
            # if (epoch + 1) % 50 == 0:
            self.valid(epoch)
            self.lr_schedule()
            self.plotter.next_epoch()
        print("train finish")

    def object_function(self, cur_h: torch.Tensor, sample_label: torch.Tensor, A: torch.Tensor, C: torch.Tensor, ind,
                        pos_ind, neg_ind, pos_label, neg_label):
        unupdated_ind = np.setdiff1d(range(self.num_train), ind)
        j1_theta_pos = 1.0 / 2 * torch.matmul(cur_h, A[pos_ind, :].t())
        j1_theta_neg = 1.0 / 2 * torch.matmul(cur_h, A[neg_ind, :].t())
        j1_theta = j1_theta_pos - j1_theta_neg - self.parameters['alpha']
        inter_loss = -torch.mean(j1_theta - torch.log(1.0 + torch.exp(j1_theta)))

        j3_theta_pos = 1.0 / 2 * torch.matmul(cur_h, C[pos_ind, :].t())
        j3_theta_neg = 1.0 / 2 * torch.matmul(cur_h, C[neg_ind, :].t())
        j3_theta = j3_theta_pos - j3_theta_neg - self.parameters['alpha']
        intra_loss = -torch.mean(j3_theta - torch.log(1.0 + torch.exp(j3_theta)))

        quantization = torch.sum(torch.pow(self.B - C, 2))
        quantization *= self.parameters['gamma']
        quantization /= (self.num_train * self.batch_size)

        balance = torch.sum(torch.pow(cur_h.t().mm(self.ones) + C[unupdated_ind].t().mm(self.ones_), 2))
        balance *= self.parameters['eta']
        balance /= (self.num_train * self.batch_size)

        graph_regularization_loss = torch.trace(self.B.t().mm(self.laplacian).mm(self.B))
        graph_regularization_loss *= self.parameters['beta']
        graph_regularization_loss /= (self.num_train * self.batch_size)

        return inter_loss, intra_loss, quantization, balance, graph_regularization_loss


def train(dataset_name: str, img_dir: str, bit: int, visdom=True, batch_size=128, cuda=True, **kwargs):
    trainer = TDH(dataset_name, img_dir, bit, visdom, batch_size, cuda, **kwargs)
    trainer.train()


def calc_decorrelation_loss(hash_code: torch.Tensor):
    miu = torch.mean(hash_code, dim=0, keepdim=True)
    C = torch.matmul(torch.mean(hash_code - miu, dim=0, keepdim=True).t(),
                     torch.mean(hash_code - miu, dim=0, keepdim=True))
    decorrelation_loss = torch.sum(torch.pow(C, 2)) - torch.sum(torch.pow(torch.diag(C), 2))
    return decorrelation_loss * 0.5


def calc_loss(B, F, G, Sim, lambdaa, gamma):
    theta = torch.matmul(F, G.transpose(0, 1)) / 2
    inter_loss = torch.sum(torch.log(1 + torch.exp(theta)) - Sim * theta)
    theta_f = torch.matmul(F, F.t()) / 2
    theta_g = torch.matmul(G, G.t()) / 2
    intra_loss_f = torch.sum(torch.log(1 + torch.exp(theta_f) - Sim * theta_f))
    intra_loss_g = torch.sum(torch.log(1 + torch.exp(theta_g) - Sim * theta_g))
    intra_loss = intra_loss_f + intra_loss_g
    miu_f = torch.mean(F, dim=0, keepdim=True)
    C_f = torch.matmul(torch.sum(F - miu_f, dim=0, keepdim=True).t(), torch.sum(F - miu_f, dim=0, keepdim=True))
    decorrelation_f = torch.sum(torch.pow(C_f, 2)) - torch.sum(torch.pow(torch.diag(C_f), 2))
    decorrelation_f *= 0.5
    miu_g = torch.mean(G, dim=0, keepdim=True)
    C_g = torch.matmul(torch.sum(G - miu_g, dim=0, keepdim=True).t(), torch.sum(G - miu_g, dim=0, keepdim=True))
    decorrelation_g = torch.sum(torch.pow(C_g, 2)) - torch.sum(torch.pow(torch.diag(C_g), 2))
    decorrelation_g *= 0.5
    decorrelation_loss = decorrelation_f + decorrelation_g
    quantization = torch.sum(torch.pow(B - F, 2) + torch.pow(B - G, 2))
    balance = torch.sum(torch.pow(F.sum(dim=0), 2) + torch.pow(G.sum(dim=0), 2))
    regularization_loss = quantization + balance
    loss = inter_loss + intra_loss + lambdaa * decorrelation_loss + gamma * regularization_loss
    return loss
