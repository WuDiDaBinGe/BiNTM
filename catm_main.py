# -*- coding: utf-8 -*-
# @Time    : 2021/7/8 下午2:16
# @Author  : WuDiDaBinGe
# @FileName: catm_main.py
# @Software: PyCharm
# -*- coding: utf-8 -*-
# @Time    : 2021/6/22 下午8:33
# @Author  : WuDiDaBinGe
# @FileName: main.py
# @Software: PyCharm
import torch
import argparse
import time
from model.atm_model import BNTM
from dataloader.dataset import DocDataset, DocNpyDataset, DataArgumentNpy
from multiprocessing import cpu_count

from model.catm_model import CATM, CBTM

parser = argparse.ArgumentParser('Bidirectional Adversarial Topic model')
parser.add_argument('--taskname', type=str, default='cnews10k', help='Taskname e.g cnews10k')
parser.add_argument('--no_below', type=int, default=5, help='The lower bound of count for words to keep, e.g 10')
parser.add_argument('--no_above', type=float, default=0.1,
                    help='The ratio of upper bound of count for words to keep, e.g 0.3')
parser.add_argument('--num_epochs', type=int, default=100,
                    help='Number of iterations (set to 100 as default, but 1000+ is recommended.)')
parser.add_argument('--n_topic', type=int, default=20, help='Num of topics')
parser.add_argument('--bkpt_continue', type=bool, default=False,
                    help='Whether to load a trained model as initialization and continue training.')
parser.add_argument('--use_tfidf', type=bool, default=True, help='Whether to use the tfidf feature for the BOW input')
parser.add_argument('--use_token', type=bool, default=False, help='dataset whether has been token')
parser.add_argument('--clean_data', type=bool, default=False, help='dataset whether has been clean in npy')
parser.add_argument('--rebuild', type=bool, default=True,
                    help='Whether to rebuild the corpus, such as tokenization, build dict etc.(default True)')
parser.add_argument('--dist', type=str, default='gmm_std',
                    help='Prior distribution for latent vectors: (dirichlet,gmm_std,gmm_ctm,gaussian etc.)')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size (default=256)')
parser.add_argument('--language', type=str, default='en', help='Dataset s language')
parser.add_argument('--lr', type=float, default=8e-4, help='learning rate')
parser.add_argument('--instance_temperature', type=float, default=0.07, help='contrastive learning temperature (0,1)')
parser.add_argument('--cluster_temperature', type=float, default=0.1, help='contrastive learning temperature (0,1)')
parser.add_argument('--auto_adj', action='store_true',
                    help='To adjust the no_above ratio automatically (default:rm top 20)')

args = parser.parse_args()


def main():
    global args
    taskname = args.taskname
    no_below = args.no_below
    no_above = args.no_above
    num_epochs = args.num_epochs
    n_topic = args.n_topic
    n_cpu = cpu_count() - 2 if cpu_count() > 2 else 2
    bkpt_continue = args.bkpt_continue
    use_tfidf = args.use_tfidf
    rebuild = args.rebuild
    dist = args.dist
    batch_size = args.batch_size
    lr = args.lr
    instance_temperature = args.instance_temperature
    cluster_temperature = args.cluster_temperature
    auto_adj = args.auto_adj
    language = args.language
    use_token = args.use_token
    clean_data = args.clean_data

    device = torch.device('cuda')
    if clean_data:
        docSet = DataArgumentNpy(taskname)
    else:
        docSet = DocDataset(taskname, no_below=no_below, no_above=no_above, rebuild=rebuild, use_tfidf=use_tfidf,
                            lang=language, use_token=use_token)
        if auto_adj:
            # no_above = docSet.topk_dfs(topk=20)
            docSet = DocDataset(taskname, lang=language, no_below=no_below, no_above=no_above, rebuild=rebuild,
                                use_tfidf=use_tfidf)

    voc_size = docSet.vob_size

    # model = CATM(bow_dim=voc_size, n_topic=n_topic, hid_dim=1024, device=device, task_name=taskname)
    # # TODO: 断点续训的时候需要改ckpt参数的路径
    # model.train_with_contra(train_data=docSet, batch_size=batch_size, test_data=docSet, epochs=num_epochs, n_critic=10, lr=lr,
    #             clean_data=clean_data, resume=bkpt_continue, gamma_temperature=instance_temperature, gamma_cluster_temperature=cluster_temperature,ckpt_path="models_save/c_atm/checkpoint_2021-07-12-22-03_20news_clean_20/ckpt_best_13500.pth")
    model = CBTM(bow_dim=voc_size, n_topic=n_topic, hid_dim=1024, device=device, task_name=taskname)
    # TODO: 断点续训的时候需要改ckpt参数的路径
    model.train_with_contra(train_data=docSet, batch_size=batch_size, test_data=docSet, epochs=num_epochs, n_critic=10,
                            lr=lr,
                            clean_data=clean_data, resume=bkpt_continue, gamma_temperature=instance_temperature, ckpt_path='models_save/c_atm_discriminator/checkpoint_2021-07-14-21-54_20news_clean_20/ckpt_best_19000.pth')
    topic_words = model.show_topic_words()
    print('\n'.join([str(lst) for lst in topic_words]))
    save_name = f'./ckpt/c_atm_discriminator_{taskname}_tp{n_topic}_{time.strftime("%Y-%m-%d-%H-%M", time.localtime())}.ckpt'
    torch.save({'generator': model.generator.state_dict(), 'encoder': model.encoder.state_dict(),
                'discriminator': model.discriminator.state_dict()}, save_name)


if __name__ == "__main__":
    main()
