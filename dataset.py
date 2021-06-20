# -*- coding: utf-8 -*-
# @Time    : 2021/6/20 下午12:53
# @Author  : WuDiDaBinGe
# @FileName: dataset.py
# @Software: PyCharm
from torch.utils.data import Dataset, DataLoader
from gensim.corpora import Dictionary
from gensim.models import  TfidfModel

class DocDataset(Dataset):
    def __init__(self):
        super(DocDataset, self).__init__()

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass
