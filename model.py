# -*- coding: utf-8 -*-
# @Time    : 2021/6/19 上午11:03
# @Author  : WuDiDaBinGe
# @FileName: model.py
# @Software: PyCharm
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from gan import Generator,Encoder,Discriminator

# Bi Neural Topic Model
class Bintm:
    def __init__(self, n_topic, bow_dim, hid_dim, device=None, task_name=None):
        super(Bintm, self).__init__()
        self.n_topic = n_topic
        self.v_dim = bow_dim
        self.hid_dim = hid_dim
        self.id2token = None
        self.task_name = task_name

        self.encoder = Encoder(v_dim=bow_dim, hid_dim=hid_dim, n_topic=n_topic)
        self.generator = Generator(v_dim=bow_dim,hid_dim=hid_dim, n_topic=n_topic)
        self.discriminator = Discriminator(v_dim=bow_dim, hid_dim=hid_dim, n_topic=n_topic)

        if device is not None:
            self.generator = self.generator.to(device)
            self.encoder = self.encoder.to(device)
            self.discriminator = self.discriminator.to(device)

    def train(self, train_data, batch_size=256, lr=1e-4, test_data=None,epochs=100):
        self.generator.train()
        self.encoder.train()
        self.discriminator.train()
        self.id2token = {v: k for k,v in train_data.dictionary.token2id.items()}

