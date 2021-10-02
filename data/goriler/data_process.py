# -*- coding: utf-8 -*-
# @Time    : 2021/9/26 下午11:47
# @Author  : WuDiDaBinGe
# @FileName: data_process.py
# @Software: PyCharm
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

articles = []

with open('grolier15276.csv', 'r') as f:
    for line in f:
        line = line.strip().split(',')
        if line[1] != "":
            line = line[1:]
            # 转成词频向量
            bow = np.zeros(15276)
            for index in range(0, len(line) - 1, 2):
                bow[int(line[index]) - 1] = int(line[index+1])
            articles.append(bow)
print(len(articles))
articles = np.array(articles)
transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
tfidf = transformer.fit_transform(articles)  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
print("tf-idf")
print(tfidf.toarray())
