# -*- coding: utf-8 -*-
# @Time    : 2021/6/19 下午1:35
# @Author  : WuDiDaBinGe
# @FileName: gan.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, n_topic, hid_dim, v_dim):
        super(Generator,self).__init__()
