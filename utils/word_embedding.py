# -*- coding: utf-8 -*-
# @Time    : 2021/9/28 下午3:09
# @Author  : WuDiDaBinGe
# @FileName: word_embedding.py
# @Software: PyCharm
import os
import pickle

import torchtext
import torch
import torchtext.vocab as vocab

cache_dir = "/home/yxb/Documents/NLP/DataSet/Glove"
glove = vocab.GloVe(name='6B', dim=300, cache=cache_dir)


def knn(W, x, k):
    # 添加的1e-9是为了数值稳定性
    cos = torch.matmul(W, x.view((-1,))) / (
            (torch.sum(W * W, dim=1) + 1e-9).sqrt() * torch.sum(x * x).sqrt())
    _, topk = torch.topk(cos, k=k)
    topk = topk.cpu().numpy()
    return topk, [cos[i].item() for i in topk]


def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.vectors,
                    embed.vectors[embed.stoi[query_token]], k + 1)
    for i, c in zip(topk[1:], cos[1:]):  # 除去输入词
        print('cosine sim=%.3f: %s' % (c, (embed.itos[i])))


def generator_dataset_embedding(dataset_vob, task_name):
    vob_size = len(dataset_vob)
    remain = []
    embedding = torch.rand((vob_size, glove.dim))
    for vob, index in dataset_vob.items():
        if vob in glove.stoi.keys():
            embedding[index] = glove.vectors[glove.stoi[vob]]
        else:
            remain.append(vob)
    torch.save(embedding, f'{task_name}_{glove.dim}_r{len(remain)}_embedding.pt')
    print(remain)


if __name__ == '__main__':
    # 没有的 ['coelenterates', 'fuhn', 'kans', 'nebr', 'nonmetropolitan', 'repr', 'sawtimber']
    dictionary_token2id = {}
    num_index = 0
    # 加载词典
    with open(os.path.join('../data/goriler', 'grolier15276_words.txt'), 'r') as f:
        for line in f:
            dictionary_token2id[line.strip()] = num_index
            num_index += 1
    generator_dataset_embedding(dictionary_token2id, 'goriler')
    # 没有的 ['oname', 'colormap', 'ripem', '_eos_', 'scsus', 'bhj', 'xterm', 'imho']
    # vocab = '../data/20news_clean/vocab.pkl'
    # vocab = pickle.load(open(vocab, 'rb'))
    # generator_dataset_embedding(vocab, '20news_clean')
