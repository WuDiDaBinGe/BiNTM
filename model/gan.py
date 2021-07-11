# -*- coding: utf-8 -*-
# @Time    : 2021/6/19 下午1:35
# @Author  : WuDiDaBinGe
# @FileName: gan.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F


def block(in_feat, out_feat, normalize=True):
    layers = [nn.Linear(in_feat, out_feat, bias=True)]
    if normalize:
        layers.append(nn.BatchNorm1d(out_feat))
    layers.append(nn.LeakyReLU(inplace=True))
    return layers


class Generator(nn.Module):
    def __init__(self, n_topic, hid_dim, v_dim):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            *block(n_topic, hid_dim),
            nn.Linear(hid_dim, v_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        output = self.generator(inputs)
        return output

    def inference(self, theta):
        return self.generator(theta)


class Encoder(nn.Module):
    def __init__(self, v_dim, hid_dim, n_topic):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            *block(v_dim, hid_dim),
            nn.Linear(hid_dim, n_topic),
            nn.Softmax(dim=1),
        )

    def forward(self, inputs):
        output = self.encoder(inputs)
        return output


class Discriminator(nn.Module):
    def __init__(self, n_topic, hid_dim, v_dim):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            * block(n_topic + v_dim, hid_dim),
            nn.Linear(hid_dim, 1),
        )

    def forward(self, reps):
        score = self.discriminator(reps)
        return score
