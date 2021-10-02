# -*- coding: utf-8 -*-
# @Time    : 2021/6/20 下午12:53
# @Author  : WuDiDaBinGe
# @FileName: dataset.py
# @Software: PyCharm
import copy
import os
import numpy as np
import torch
import gensim
from torch.utils.data import Dataset, DataLoader
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
from tokenization import *
from utils.tf_idf_data_argument import get_data_stats, TfIdfWordRep


def onehot(data, min_length):
    # return word frequent vector
    return np.bincount(data, minlength=min_length)

class DocNpyDataset(Dataset):
    def __init__(self, task_dir, use_tfidf=True, no_below=1, no_above=0.1, stopwords=None):
        super(DocNpyDataset, self).__init__()
        self.task_name = task_dir
        # file current dir
        cwd = os.path.split(os.path.realpath(__file__))[0]
        cwd = os.path.split(cwd)[0]
        text_path = os.path.join(cwd, 'data', task_dir)
        if task_dir == '20news_clean':
            dataset_tr = 'train.txt.npy'
            self.data_tr_row = np.load(os.path.join(text_path, dataset_tr), allow_pickle=True, encoding='latin1')
            self.data_tr_row = [doc for doc in self.data_tr_row if np.sum(doc) != 0 and len(doc) > 2]
            vocab_p = 'vocab.pkl'
            self.dictionary_token2id = pickle.load(open(os.path.join(text_path, vocab_p), 'rb'))
            self.vob_size = len(self.dictionary_token2id)
            # word frequent vector
            self.data_tr = np.array(
                [onehot(doc.astype('int'), self.vob_size) for doc in self.data_tr_row if np.sum(doc) != 0])

        elif task_dir == 'goriler':
            dataset_tr = 'grolier15276.csv'
            self.dictionary_token2id = {}
            num_index = 0
            # 加载词典
            with open(os.path.join(text_path, 'grolier15276_words.txt'), 'r') as f:
                for line in f:
                    self.dictionary_token2id[line.strip()] = num_index
                    num_index += 1
            self.vob_size = len(self.dictionary_token2id)
            articles = []  # 存放词频向量
            with open(os.path.join(text_path, dataset_tr), 'r') as f:
                for line in f:
                    line = line.strip().split(',')
                    if line[1] != "":
                        line = line[1:]
                        # 转成词频向量
                        bow = np.zeros(self.vob_size)
                        for index in range(0, len(line) - 1, 2):
                            bow[int(line[index]) - 1] = int(line[index + 1])
                        articles.append(bow)
            self.data_tr = np.array(articles)
        self.num_docs = len(self.data_tr)
        self.use_tfidf = use_tfidf
        if self.use_tfidf:
            transformer = TfidfTransformer()  # get tf-idf weight
            self.tfidf = transformer.fit_transform(self.data_tr)
            self.tfidf = self.tfidf.toarray()
        self.dictionary_id2token = {v: k for k, v in self.dictionary_token2id.items()}

        print(f'Processed {self.num_docs} documents')

    def __getitem__(self, idx):
        if self.use_tfidf:
            return self.tfidf[idx]
        else:
            return self.data_tr[idx]

    def __len__(self):
        return self.num_docs

class DocDataset(Dataset):
    def __init__(self, task_name, lang='zh', text_path=None, use_tfidf=True, no_below=1, no_above=0.1, stopwords=None,
                 rebuild=True, use_token=False):
        super(DocDataset, self).__init__()

        cwd = os.getcwd()
        text_path = os.path.join(cwd, '../data', f'{task_name}_lines.txt') if text_path is None else text_path
        tmp_dir = os.path.join(cwd, '../data', task_name)
        self.txt_lines = [line.strip() for line in open(text_path, 'r')]
        self.use_tfidf = use_tfidf
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        if not rebuild and os.path.exists(os.path.join(tmp_dir, 'corpus.mm')):
            self.bows = gensim.corpora.MmCorpus(os.path.join(tmp_dir, 'corpus.mm'))
            if self.use_tfidf:
                self.tfidf = gensim.corpora.MmCorpus(os.path.join(tmp_dir, 'tfidf.mm'))
            self.dictionary = Dictionary.load_from_text(os.path.join(tmp_dir, 'dict.txt'))
            self.docs = pickle.load(open(os.path.join(tmp_dir, 'docs.pkl'), 'rb'))
            self.dictionary.id2token = {v: k for k, v in self.dictionary.token2id.items()}
        else:
            if stopwords is None:
                stopwords = set([l.strip() for l in open(os.path.join(cwd, '../data', 'stopwords'), 'r')])
            print('Tokenizing...')
            if use_token:
                self.docs = [line.split(' ') for line in self.txt_lines]
                self.docs = [line for line in self.docs if line != []]
            else:
                tokenizer = globals()[LANG_CLS[lang]](stopwords=stopwords)
                # docs after tokenizer
                self.docs = tokenizer.tokenize(self.txt_lines)
                self.docs = [line for line in self.docs if line != []]

            # build dictionary
            self.dictionary = Dictionary(self.docs)
            self.dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_tokens=None)
            self.dictionary.compactify()
            self.dictionary.id2token = {v: k for k, v in self.dictionary.token2id.items()}
            self.bows, _docs = [], []
            print("docs  " + str(len(self.docs)))
            for doc in self.docs:
                _bow = self.dictionary.doc2bow(doc)
                if len(_bow):
                    self.bows.append(_bow)
                    _docs.append(list(doc))
            print("bows  1 " + str(len(self.bows)))
            # remove wrong word
            self.docs = _docs
            print("bows  2 " + str(len(self.bows)))
            if self.use_tfidf:
                self.tfidf_model = TfidfModel(self.bows)
                self.tfidf = [self.tfidf_model[bow] for bow in self.bows]

            # serialize the dictionary
            gensim.corpora.MmCorpus.serialize(os.path.join(tmp_dir, "corpus.mm"), self.bows)
            self.dictionary.save_as_text(os.path.join(tmp_dir, 'dict.txt'))
            pickle.dump(self.docs, open(os.path.join(tmp_dir, 'docs.pkl'), 'wb'))
            if self.use_tfidf:
                gensim.corpora.MmCorpus.serialize(os.path.join(tmp_dir, 'tfidf.mm'), self.tfidf)
        self.vob_size = len(self.dictionary)
        self.num_docs = len(self.bows)
        self.dictionary_id2token = {v: k for k, v in self.dictionary.token2id.items()}
        print(f'Processed {len(self.bows)} documents')

    def __getitem__(self, idx):
        bow = torch.zeros(self.vob_size)
        # item [[token_id1,token_id2,...],[weight,weight,..]]
        if self.use_tfidf:
            item = list(zip(*self.tfidf[idx]))
        else:
            item = list(zip(*self.bows[idx]))
        bow[list(item[0])] = torch.tensor(list(item[1])).float()
        txt = self.docs[idx]
        return txt, bow

    def __len__(self):
        return self.num_docs

    def collate_fn(self, batch_data):
        texts, bows = list(zip(*batch_data))
        return texts, torch.stack(bows, dim=0)

    def __iter__(self):
        for doc in self.docs:
            yield doc





class DataArgumentNpy(DocNpyDataset):
    def __init__(self, task_dir, use_tfidf=True):
        super(DataArgumentNpy, self).__init__(task_dir, use_tfidf)
        self.data_argument_1 = copy.deepcopy(self.data_tr_row)
        self.data_argument_2 = copy.deepcopy(self.data_tr_row)
        self.data_status = get_data_stats(self.data_tr_row)
        self.op = TfIdfWordRep(0.6, self.data_status)
        # data argument by tf idf
        for i in range(len(self.data_tr)):
            self.op(self.data_argument_1[i])
            self.op(self.data_argument_2[i])
        # word frequent vector
        self.data_tr_1 = np.array(
            [onehot(doc.astype('int'), self.vob_size) for doc in self.data_argument_1 if np.sum(doc) != 0])
        self.data_tr_2 = np.array(
            [onehot(doc.astype('int'), self.vob_size) for doc in self.data_argument_2 if np.sum(doc) != 0])
        if self.use_tfidf:
            transformer = TfidfTransformer()  # get tf-idf weight
            self.tfidf_1 = transformer.fit_transform(self.data_tr_1)
            self.tfidf_1 = self.tfidf_1.toarray()
            self.tfidf_2 = transformer.fit_transform(self.data_tr_2)
            self.tfidf_2 = self.tfidf_2.toarray()

    def __getitem__(self, idx):
        if self.use_tfidf:
            return self.tfidf[idx], self.tfidf_1[idx], self.tfidf_2[idx]
        else:
            return self.data_tr[idx], self.data_tr_1[idx], self.data_tr_2[idx]


if __name__ == '__main__':
    # docSet = DocDataset('military')
    # dataloader = DataLoader(docSet, batch_size=64, shuffle=True, collate_fn=docSet.collate_fn)
    # print('docSet.docs[10]:', docSet.docs[10])
    # print(next(iter(dataloader)))
    # for iter, data in enumerate(dataloader):
    #     txts, bows_real = data
    #     print(bows_real.shape)
    #     print(bows_real)
    #     # normalize weights
    #     bows_real /= torch.sum(bows_real, dim=1, keepdim=True)
    #     print(bows_real.shape)

    docSet_npy = DocNpyDataset('20news_clean')
    print(docSet_npy.vob_size)
    dataloader = DataLoader(docSet_npy, batch_size=64, shuffle=True)
    print(np.sum(docSet_npy.tfidf[10]))
    print(next(iter(dataloader)).shape)

    # docSet_npy = DataArgumentNpy('20news_clean')
    # dataloader = DataLoader(docSet_npy, batch_size=64, shuffle=True)
    # print('docSet.docs[10]:', docSet_npy.tfidf[10])
    # print(next(iter(dataloader))[0].shape)
