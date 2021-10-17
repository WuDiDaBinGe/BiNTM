# -*- coding: utf-8 -*-
# @Time    : 2021/6/30 下午2:50
# @Author  : WuDiDaBinGe
# @FileName: interface.py
# @Software: PyCharm
import torch
import argparse
from model.atm_model import BNTM
from dataloader.dataset import DocDataset, DocNpyDataset
parser = argparse.ArgumentParser('Bidirectional Adversarial Topic model')
parser.add_argument('--taskname', type=str, default='cnews10k', help='Taskname e.g cnews10k')
parser.add_argument('--no_below', type=int, default=5, help='The lower bound of count for words to keep, e.g 10')
parser.add_argument('--no_above', type=float, default=0.1,
                    help='The ratio of upper bound of count for words to keep, e.g 0.3')
parser.add_argument('--n_topic', type=int, default=20, help='Num of topics')
parser.add_argument('--ckpt_path', type=str, default='./ckpt',
                    help='Whether to load a trained model as initialization and continue training.')
parser.add_argument('--use_tfidf', type=bool, default=True, help='Whether to use the tfidf feature for the BOW input')
parser.add_argument('--use_token', type=bool, default=False, help='dataset whether has been token')
parser.add_argument('--clean_data', type=bool, default=False, help='dataset whether has been clean in npy')
parser.add_argument('--rebuild', type=bool, default=True,
                    help='Whether to rebuild the corpus, such as tokenization, build dict etc.(default True)')
parser.add_argument('--language', type=str, default='en', help='Dataset s language')

args = parser.parse_args()


def main():
    global args
    taskname = args.taskname
    no_below = args.no_below
    no_above = args.no_above
    n_topic = args.n_topic
    ckpt_path = args.ckpt_path
    use_tfidf = args.use_tfidf
    language = args.language
    use_token = args.use_token
    clean_data = args.clean_data
    rebuild = args.rebuild
    device = torch.device('cuda')
    if clean_data:
        docSet = DocNpyDataset(taskname)
    else:
        docSet = DocDataset(taskname, no_below=no_below, no_above=no_above, rebuild=rebuild, use_tfidf=use_tfidf, lang=language, use_token=use_token)

    voc_size = docSet.vob_size

    model = BNTM(bow_dim=voc_size, n_topic=n_topic, hid_dim=1024, device=device, task_name=taskname)
    # TODO:修改路径
    model.init_by_checkpoints(ckpt_path=ckpt_path)
    model.interface_topic_words(clean_data=clean_data, test_data=docSet)


if __name__ == "__main__":
    main()
