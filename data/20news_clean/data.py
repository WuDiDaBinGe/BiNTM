import numpy as np
import pickle


def onehot(data, min_length):
    # index appear how much times
    return np.bincount(data, minlength=min_length)


dataset_tr = 'train.txt.npy'
data_tr = np.load(dataset_tr, allow_pickle=True, encoding='latin1')
print("Train length:" + str(len(data_tr)))
dataset_te = 'test.txt.npy'
data_te = np.load(dataset_te, allow_pickle=True, encoding='latin1')
print("Test length:" + str(len(data_te)))
vocab = 'vocab.pkl'
vocab = pickle.load(open(vocab, 'rb'))
vocab_size = len(vocab)
print(vocab_size, vocab)
# --------------convert to one-hot representation------------------
print("Converting data to one-hot representation")
data_tr = np.array([onehot(doc.astype('int'), vocab_size) for doc in data_tr if np.sum(doc) != 0])
data_te = np.array([onehot(doc.astype('int'), vocab_size) for doc in data_te if np.sum(doc) != 0])

import os
import sys
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
tfidf = transformer.fit_transform(data_tr)  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
print("tf-idf")
print(tfidf.toarray())
tf_idf = tfidf.toarray()
# corpus = ["我 来到 北京 清华大学",  # 第一类文本
#           # 切词后的结果，词之间以空格隔开
#           "他 来到 了 网易 杭研 大厦",  # 第二类文本的切词结果
#           "小明 硕士 毕业 与 中国 科学院",  # 第三类文本的切词结果
#           "我 爱 北京 天安门"]  # 第四类文本的切词结果
# vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
#
# word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
# weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
# for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
#     print(u"-------这里输出第", i, u"类文本的词语tf-idf权重------")
#     for j in range(len(word)):
#         print(word[j], weight[i][j])
import torch

tf_idf = torch.tensor(tf_idf, dtype=torch.double)
print("*************************")
embedding = torch.load(f'./20news_clean_300_embedding.pt').double()
print(tf_idf.shape)
print(embedding.shape)

tf_idf /= torch.sum(tf_idf, dim=1, keepdim=True)
x_emb = torch.mm(tf_idf, embedding)

print(x_emb)
