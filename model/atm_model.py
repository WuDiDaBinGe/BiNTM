# -*- coding: utf-8 -*-
# @Time    : 2021/6/19 上午11:03
# @Author  : WuDiDaBinGe
# @FileName: model.py
# @Software: PyCharm
import os
import threading
import time
from typing import List, Any

from palmettopy.palmetto import Palmetto
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from model.gan import Generator, Encoder, Discriminator
from utils.caculate_coherence import get_coherence
from utils.utils import evaluate_topic_quality


# Bi Neural Topic Model
class BNTM:
    def __init__(self, n_topic, bow_dim, hid_dim, device=None, task_name=None):
        super(BNTM, self).__init__()
        self.n_topic = n_topic
        self.v_dim = bow_dim
        self.hid_dim = hid_dim
        self.id2token = None
        self.task_name = task_name
        self.max_npmi_step = 0
        self.max_npmi_value = 0
        self.logdir_name = "atm"
        self.date = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        # self.writer = SummaryWriter(
        #    f'log/atm/{self.task_name}_{time.strftime("%Y-%m-%d-%H-%M", time.localtime())}_topic{self.n_topic}')
        self.writer = SummaryWriter('log/atm/20news_clean_2021-07-15-16-22_topic20')
        self.encoder = Encoder(v_dim=bow_dim, hid_dim=hid_dim, n_topic=n_topic)
        self.generator = Generator(v_dim=bow_dim, hid_dim=hid_dim, n_topic=n_topic)
        self.discriminator = Discriminator(v_dim=bow_dim, hid_dim=hid_dim, n_topic=n_topic)
        self.device = device
        if device is not None:
            self.generator = self.generator.to(device)
            self.encoder = self.encoder.to(device)
            self.discriminator = self.discriminator.to(device)

    def init_by_checkpoints(self, ckpt_path='models/checkpoint_20news_clean/ckpt_best_16500.pth'):
        checkpoint = torch.load(ckpt_path)
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.generator.load_state_dict(checkpoint['generator'])
        self.encoder.load_state_dict(checkpoint['encoder'])

    def train(self, train_data, batch_size=64, clip=0.01, lr=1e-4, test_data=None, epochs=100, beta_1=0.5,
              beta_2=0.999, n_critic=5, clean_data=False, resume=False,
              ckpt_path='models_save/atm/checkpoint_20news_clean_20/ckpt_best_1000.pth'):
        self.generator.train()
        self.encoder.train()
        self.discriminator.train()
        start_epoch = -1
        if clean_data:
            self.id2token = train_data.dictionary_id2token
            data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        else:
            self.id2token = {v: k for k, v in train_data.dictionary.token2id.items()}
            data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                     collate_fn=train_data.collate_fn)
        optim_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(beta_1, beta_2))
        optim_e = torch.optim.Adam(self.encoder.parameters(), lr=lr, betas=(beta_1, beta_2))
        optim_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta_1, beta_2))
        loss_E, loss_G, loss_D = 0, 0, 0
        if resume:
            checkpoint = torch.load(ckpt_path)
            self.discriminator.load_state_dict(checkpoint['discriminator'])
            self.generator.load_state_dict(checkpoint['generator'])
            self.encoder.load_state_dict(checkpoint['encoder'])
            optim_g.load_state_dict(checkpoint['optimizer_g'])
            optim_d.load_state_dict(checkpoint['optimizer_d'])
            optim_e.load_state_dict(checkpoint['optimizer_e'])
            start_epoch = checkpoint['epoch']
            self.max_npmi_value = checkpoint['maxValue']
            self.max_npmi_step = checkpoint['maxStep']

        for epoch in range(start_epoch + 1, epochs + 1):
            for iter_num, data in enumerate(data_loader):
                if clean_data:
                    bows_real = data
                    bows_real = bows_real.float()
                else:
                    texts, bows_real = data
                bows_real = bows_real.to(self.device)
                # train  discriminator
                optim_d.zero_grad()
                # normalize weight
                bows_real /= torch.sum(bows_real, dim=1, keepdim=True)
                # sample
                topic_fake = torch.from_numpy(np.random.dirichlet(alpha=1.0 * np.ones(self.n_topic) / self.n_topic,
                                                                  size=len(bows_real))).float().to(self.device)
                # use detach() to stop the gradient backward p(frozen the g and e parameters)

                real_p = torch.cat([self.encoder(bows_real).detach(), bows_real], dim=1)
                fake_p = torch.cat([topic_fake, self.generator(topic_fake).detach()], dim=1)
                loss_D = -torch.mean(self.discriminator(real_p)) + torch.mean(self.discriminator(fake_p))
                loss_D.backward()
                optim_d.step()
                # Clip weights of discriminator
                for p in self.discriminator.parameters():
                    p.data.clamp_(-clip, clip)
                if iter_num % n_critic == 0:
                    # train generator and encoder
                    optim_g.zero_grad()
                    p_fake = torch.cat([topic_fake, self.generator(topic_fake)], dim=1)
                    loss_G = -torch.mean(self.discriminator(p_fake))
                    loss_G.backward()
                    optim_g.step()

                    optim_e.zero_grad()
                    p_real = torch.cat([self.encoder(bows_real), bows_real], dim=1)
                    loss_E = torch.mean(self.discriminator(p_real))
                    loss_E.backward()
                    optim_e.step()
                    print(
                        f'Epoch {(epoch + 1):>3d}\tIter {(iter_num + 1):>4d}\tLoss_D:{loss_D.item():<.7f}\tLoss_G:{loss_G.item():<.7f}\tloss_E:{loss_E.item():<.7f}')

            # tensorboardX
            if epoch % 50 == 0:
                self.writer.add_scalars("Train/Loss", {"D_loss": loss_D, "E_loss": loss_E, "G_loss": loss_G},
                                        epoch)
            if epoch % 250 == 0:
                if test_data is not None:
                    c_a, c_p, npmi = self.get_topic_coherence()
                    if self.max_npmi_value < npmi:
                        self.max_npmi_value = npmi
                        self.max_npmi_step = epoch
                        self.save_model_ckpt(optim_g=optim_g, optim_e=optim_e, optim_d=optim_d, epoch=epoch,
                                             max_step=self.max_npmi_step, max_value=self.max_npmi_value)
                    print(c_a, c_p, npmi)
                    print("max_epoch:" + str(self.max_npmi_step) + "  max_value:" + str(self.max_npmi_value))
                    self.writer.add_scalars("Topic Coherence", {'c_a': c_a, 'c_p': c_p, 'npmi': npmi},
                                            epoch)

    def save_model_ckpt(self, optim_g, optim_e, optim_d, epoch, max_step, max_value):
        checkpoint = {'generator': self.generator.state_dict(),
                      'encoder': self.encoder.state_dict(),
                      'discriminator': self.discriminator.state_dict(),
                      'optimizer_g': optim_g.state_dict(),
                      'optimizer_e': optim_e.state_dict(),
                      'optimizer_d': optim_d.state_dict(),
                      'epoch': epoch,
                      'maxStep': max_step,
                      'maxValue': max_value
                      }
        if not os.path.isdir(f"models_save/{self.logdir_name}/checkpoint_{self.task_name}_{self.n_topic}_{self.date}"):
            os.mkdir(f"models_save/{self.logdir_name}/checkpoint_{self.task_name}_{self.n_topic}_{self.date}")
        torch.save(checkpoint,
                   f'models_save/{self.logdir_name}/checkpoint_{self.task_name}_{self.n_topic}_{self.date}/ckpt_best.pth')

    def show_topic_words(self, topic_id=None, topK=10):
        with torch.no_grad():
            topic_words = []
            idxes = torch.eye(self.n_topic).to(self.device)
            word_dist = self.generator.inference(idxes)
            vals, indices = torch.topk(word_dist, topK, dim=1)
            vals = vals.cpu().tolist()
            indices = indices.cpu().tolist()
            if topic_id is None:
                for i in range(self.n_topic):
                    topic_words.append([self.id2token[idx] for idx in indices[i]])
            else:
                topic_words.append([self.id2token[idx] for idx in indices[topic_id]])
            return topic_words

    def evaluate(self, test_data, calc4each=False):
        topic_words = self.show_topic_words()
        print('\n'.join([str(lst) for lst in topic_words]))
        return evaluate_topic_quality(topic_words, test_data, taskname=self.task_name, calc4each=calc4each)

    def get_topic_coherence(self):
        topic_words = self.show_topic_words()
        print('\n'.join([str(lst) for lst in topic_words]))
        return get_coherence(topic_words)

    def interface_topic_words(self, clean_data=True, test_data=None):
        self.id2token = test_data.dictionary_id2token
        c_a, c_p, npmi = self.get_topic_coherence()
        print(f'c_a:{c_a},c_p:{c_p}, npmi:{npmi}')
