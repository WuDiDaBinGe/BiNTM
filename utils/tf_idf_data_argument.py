# -*- coding: utf-8 -*-
# @Time    : 2021/7/7 上午10:43
# @Author  : WuDiDaBinGe
# @FileName: tf_idf_data_argument.py
# @Software: PyCharm
import collections
import copy
import json
import math
import torch
import os
import pickle
import string
from absl import flags
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

FLAGS = flags.FLAGS

printable = set(string.printable)


def filter_unicode(st):
    return "".join([c for c in st if c in printable])


class EfficientRandomGen(object):
    """A base class that generate multiple random numbers at the same time."""

    def reset_random_prob(self):
        """Generate many random numbers at the same time and cache them."""
        cache_len = 100000
        self.random_prob_cache = np.random.random(size=(cache_len,))
        self.random_prob_ptr = cache_len - 1

    def get_random_prob(self):
        """Get a random number."""
        value = self.random_prob_cache[self.random_prob_ptr]
        self.random_prob_ptr -= 1
        if self.random_prob_ptr == -1:
            self.reset_random_prob()
        return value

    def get_random_token(self):
        """Get a random token."""
        token = self.token_list[self.token_ptr]
        self.token_ptr -= 1
        if self.token_ptr == -1:
            self.reset_token_list()
        return token


def get_data_stats(examples):
    """Compute the IDF score for each word. Then compute the TF-IDF score. on the whole corpus"""
    word_doc_freq = collections.defaultdict(int)
    # Compute IDF
    for i in range(len(examples)):
        cur_word_dict = {}
        cur_sent = copy.deepcopy(examples[i])
        for word in cur_sent:
            cur_word_dict[word] = 1
        for word in cur_word_dict:
            word_doc_freq[word] += 1
    idf = {}
    for word in word_doc_freq:
        idf[word] = math.log(len(examples) * 1. / word_doc_freq[word])
    # Compute TF-IDF
    tf_idf = {}
    for i in range(len(examples)):
        cur_word_dict = {}
        cur_sent = copy.deepcopy(examples[i])
        for word in cur_sent:
            if word not in tf_idf:
                tf_idf[word] = 0
            tf_idf[word] += 1. / len(cur_sent) * idf[word]
    return {
        "idf": idf,
        "tf_idf": tf_idf,
    }


class TfIdfWordRep(EfficientRandomGen):
    """TF-IDF Based Word Replacement."""

    def __init__(self, token_prob, data_stats):
        super(TfIdfWordRep, self).__init__()
        self.token_prob = token_prob
        self.data_stats = data_stats
        self.idf = data_stats["idf"]
        self.tf_idf = data_stats["tf_idf"]
        data_stats = copy.deepcopy(data_stats)
        tf_idf_items = data_stats["tf_idf"].items()
        tf_idf_items = sorted(tf_idf_items, key=lambda item: -item[1])
        self.tf_idf_keys = []
        self.tf_idf_values = []
        for key, value in tf_idf_items:
            self.tf_idf_keys += [key]
            self.tf_idf_values += [value]

        self.normalized_tf_idf = np.array(self.tf_idf_values)
        self.normalized_tf_idf = (self.normalized_tf_idf.max() - self.normalized_tf_idf)
        self.normalized_tf_idf = (self.normalized_tf_idf / self.normalized_tf_idf.sum())
        self.reset_token_list()
        self.reset_random_prob()

    def get_replace_prob(self, all_words):
        """Compute the probability of replacing tokens in a sentence."""
        cur_tf_idf = collections.defaultdict(int)
        # compute current tf idf
        for word in all_words:
            cur_tf_idf[word] += 1. / len(all_words) * self.idf[word]
        replace_prob = []
        for word in all_words:
            replace_prob += [cur_tf_idf[word]]
        replace_prob = np.array(replace_prob)
        replace_prob = np.max(replace_prob) - replace_prob
        replace_prob = (replace_prob / replace_prob.sum() * self.token_prob * len(all_words))
        return replace_prob

    def __call__(self, example):
        if self.get_random_prob() < 0.001:
            show_example = True
        else:
            show_example = False
        all_words = copy.deepcopy(example)
        # if show_example:
        #     print("before tf_idf_unif aug: {:s}".format(str(all_words)))
        replace_prob = self.get_replace_prob(all_words)
        example = self.replace_tokens(
            example,
            replace_prob[:len(example)]
        )
        # if show_example:
        #     all_words = copy.deepcopy(example)
        #     print("after tf_idf_unif aug: {:s}".format(str(all_words)))
        return example

    def replace_tokens(self, word_list, replace_prob):
        """Replace tokens in a sentence."""
        for i in range(len(word_list)):
            if self.get_random_prob() < replace_prob[i]:
                word_list[i] = self.get_random_token()
        return word_list

    def reset_token_list(self):
        cache_len = len(self.tf_idf_keys)
        token_list_idx = np.random.choice(
            cache_len, (cache_len,), p=self.normalized_tf_idf)
        self.token_list = []
        for idx in token_list_idx:
            self.token_list += [self.tf_idf_keys[idx]]
        self.token_ptr = len(self.token_list) - 1
        # print("sampled token list: {:s}".format(
        #     filter_unicode(" ".join(str(self.token_list)))))


def use_tfidf_del_weight(tfidf_weights, len_factor):
    non_zero_num = len(torch.nonzero(tfidf_weights))
    zero_num = len(tfidf_weights) - non_zero_num
    wait_index = torch.argsort(tfidf_weights)[zero_num:zero_num + int(non_zero_num * len_factor)]
    rand_prob_cache = torch.rand(int(non_zero_num * len_factor) + 2)
    for i in range(len(wait_index)):
        prob = 0.1 + i * (0.9 - 0.1) / len(wait_index)
        if prob < rand_prob_cache[i]:
            tfidf_weights[wait_index[i]] = 0
    return tfidf_weights


def batch_argument_del_tfidf(tfidf_weights_matrix, len_factor):
    tfidf_weights_matrix_argument = copy.deepcopy(tfidf_weights_matrix)
    for i in range(len(tfidf_weights_matrix_argument)):
        tfidf_weights_matrix_argument[i] = use_tfidf_del_weight(tfidf_weights_matrix_argument[i],len_factor)
    return tfidf_weights_matrix_argument


def onehot(data, min_length):
    return np.bincount(data, minlength=min_length)


if __name__ == '__main__':
    cwd = os.getcwd()
    text_path = os.path.join('../', 'data', "20news_clean")
    dataset_tr = 'train.txt.npy'
    data_tr = np.load(os.path.join(text_path, dataset_tr), allow_pickle=True, encoding='latin1')
    data_raw_1 = copy.deepcopy(data_tr)
    data_raw_2 = copy.deepcopy(data_tr)
    vocab_p = 'vocab.pkl'
    dictionary_token2id = pickle.load(open(os.path.join(text_path, vocab_p), 'rb'))
    vob_size = len(dictionary_token2id)
    data_tr = np.array(
        [onehot(doc.astype('int'), vob_size) for doc in data_tr if np.sum(doc) != 0])
    transformer = TfidfTransformer()  # get tf-idf weight
    tfidf = transformer.fit_transform(data_tr)
    tfidf = tfidf.toarray()
    # data_status = get_data_stats(data_tr)
    # op = TfIdfWordRep(0.7, data_status)
    # x = (data_tr[0] == op(data_raw_1[0]))
    # y = (data_tr[0] == op(data_raw_2[0]))
    # print(np.where(x == True))
    # print(np.where(y == True))
    # print(np.bincount(x))
    # print(np.bincount(y))
    tfidf = tfidf / np.sum(tfidf, axis=1, keepdims=True)
    tfidf = torch.tensor(tfidf)
    tfidf = torch.rand((256, 1999))
    print(tfidf)
    data_argument_matrix = batch_argument_del_tfidf(tfidf, 0.3)
    print(data_argument_matrix)
    # print(len(np.nonzero(data_argument_matrix[0])[0]))
    # print(len(np.nonzero(tfidf[0])[0]))
