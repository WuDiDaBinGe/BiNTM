# -*- coding: utf-8 -*-
# @Time    : 2021/8/19 下午8:41
# @Author  : WuDiDaBinGe
# @FileName: config.py
# @Software: PyCharm
import time

import torch
import gensim
import numpy as np
from tensorboardX import SummaryWriter


class Topic_Config(object):
    def __init__(self, dataset):
        self.vocab_path = '../dataset/vocab.txt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.train_path = dataset + '/datagrand_2021_train.csv'  # 训练集
        self.test_path = dataset + '/datagrand_2021_test.csv'
        self.log_path = dataset + '/log/wtm/'
        self.writer = SummaryWriter(self.log_path + time.strftime('%m-%d_%H.%M', time.localtime()))
        self.save_path = dataset + '/saved_dict/wae/' + time.strftime('%m-%d_%H.%M',
                                                                      time.localtime()) + '.ckpt'  # 模型训练结果
        # train's hyper parameter
        self.n_topic = 10
        self.batch_size = 512
        self.epoch = 10000
        self.lr = 1e-3
        self.dist = 'gmm-ctm'
        self.beta = 1.0
        self.dropout = 0.0
