# -*- coding: utf-8 -*-
# @Time    : 2021/6/20 下午12:53
# @Author  : WuDiDaBinGe
# @FileName: dataset.py
# @Software: PyCharm
import os
import torch
import gensim
from torch.utils.data import Dataset, DataLoader
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import pickle

from tokenization import *


class DocDataset(Dataset):
    def __init__(self, task_name, text_path=None, use_tfidf=False, no_below=1, no_above=0.1, stopwords=None,
                 rebuild=True, lang='zh'):
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
                stopwords = set([l.strip() for l in open(os.path.join(cwd, 'data', 'stopwords'), 'r')])
            print('Tokenizing...')
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


if __name__ == '__main__':
    docSet = DocDataset('military')
    dataloader = DataLoader(docSet, batch_size=64, shuffle=True, collate_fn=docSet.collate_fn)
    print('docSet.docs[10]:', docSet.docs[10])
    print(next(iter(dataloader)))
    for iter, data in enumerate(dataloader):
        txts, bows_real = data
        print(bows_real.shape)
        # normalize weights
        bows_real /= torch.sum(bows_real, dim=1, keepdim=True)
        print(bows_real.shape)