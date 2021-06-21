# -*- coding: utf-8 -*-
# @Time    : 2021/6/20 下午12:53
# @Author  : WuDiDaBinGe
# @FileName: dataset.py
# @Software: PyCharm
import os

import gensim
from torch.utils.data import Dataset, DataLoader
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import pickle

from tokenization import LANG_CLS


class DocDataset(Dataset):
    def __init__(self, text_path, task_name, use_tfidf, no_below=1, no_above=0.1, stopwords=None, rebuild=False,
                 lang='en'):
        super(DocDataset, self).__init__()
        cwd = os.getcwd()
        text_path = os.path.join(cwd, 'data', f'{task_name}_lines.txt') if text_path is None else text_path
        tmp_dir = os.path.join(cwd, 'data', task_name)
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
                stopwords = set([l.strip() for l in open(os.path.join(cwd, 'data', 'stopwords.txt'), 'r')])
            print('Tokenizing...')
            tokenizer = globals()[LANG_CLS[lang]](stopwords=stopwords)
            self.docs = tokenizer.tokenize(self.txt_lines)
            self.docs = [line for line in self.docs if line != []]
            # build dictionary
            self.dictionary = Dictionary(self.docs)
            self.dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_tokens=None)
            self.dictionary.compactify()
            self.dictionary.id2token = {v: k for k, v in self.dictionary.token2id.items()}
            self.bows, _docs = [], []
            for doc in self.docs:
                _bow = self.dictionary.doc2bow(doc)
                if _bow:
                    _docs.append(list(doc))
                    self.bows.append(_bow)

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


if __name__ == '__main__':
    globals()[LANG_CLS['en']](stopwords=stopwords)
