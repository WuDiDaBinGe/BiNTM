# -*- coding: utf-8 -*-
# @Time    : 2021/6/19 上午11:03
# @Author  : WuDiDaBinGe
# @FileName: model.py
# @Software: PyCharm
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from gan import Generator, Encoder, Discriminator
from utils import evaluate_topic_quality

writer = SummaryWriter('log')


# Bi Neural Topic Model
class BNTM:
    def __init__(self, n_topic, bow_dim, hid_dim, device=None, task_name=None):
        super(BNTM, self).__init__()
        self.n_topic = n_topic
        self.v_dim = bow_dim
        self.hid_dim = hid_dim
        self.id2token = None
        self.task_name = task_name

        self.encoder = Encoder(v_dim=bow_dim, hid_dim=hid_dim, n_topic=n_topic)
        self.generator = Generator(v_dim=bow_dim, hid_dim=hid_dim, n_topic=n_topic)
        self.discriminator = Discriminator(v_dim=bow_dim, hid_dim=hid_dim, n_topic=n_topic)
        self.device = device
        if device is not None:
            self.generator = self.generator.to(device)
            self.encoder = self.encoder.to(device)
            self.discriminator = self.discriminator.to(device)

    def train(self, train_data, batch_size=64, clip=0.01, lr=1e-4, test_data=None, epochs=100, beta_1=0.5,
              beta_2=0.999, n_critic=5):
        self.generator.train()
        self.encoder.train()
        self.discriminator.train()
        self.id2token = {v: k for k, v in train_data.dictionary.token2id.items()}
        data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                 collate_fn=train_data.collate_fn)
        optim_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(beta_1, beta_2))
        optim_e = torch.optim.Adam(self.encoder.parameters(), lr=lr, betas=(beta_1, beta_2))
        optim_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta_1, beta_2))
        loss_E, loss_G, loss_D = 0, 0, 0
        for epoch in range(epochs):
            for iter_num, data in enumerate(data_loader):
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
            if epoch % 5 == 0:
                writer.add_scalars("Train/Loss", {"D_loss": loss_D, "E_loss": loss_E, "G_loss": loss_G},
                                   epoch)
                if test_data is not None:
                    c_v, c_w2v, c_uci, c_npmi, mimno_tc, td = self.evaluate(test_data, calc4each=False)
                    writer.add_scalars("Topic Coherence",
                                       {'c_v': c_v, 'c_uci': c_uci, 'c_npmi': c_npmi},
                                       epoch)

    def show_topic_words(self, topic_id=None, topK=15):
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
