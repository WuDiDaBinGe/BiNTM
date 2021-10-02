# -*- coding: utf-8 -*-
# @Time    : 2021/9/25 下午3:36
# @Author  : WuDiDaBinGe
# @FileName: gan-qp-tm.py
# @Software: PyCharm
import time
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from model.atm_model import BNTM


class Gan_QP(BNTM):
    def __init__(self, n_topic, bow_dim, hid_dim, device=None, task_name=None):
        super(Gan_QP, self).__init__(n_topic, bow_dim, hid_dim, device=None, task_name=None)
        self.date = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        self.writer = SummaryWriter(
            f'log/Gan_QP/{self.task_name}_{time.strftime("%Y-%m-%d-%H-%M", time.localtime())}_topic{self.n_topic}')

    def train(self, train_data, batch_size=64, clip=0.01, lr=1e-4, test_data=None, epochs=100, beta_1=0.5,
              beta_2=0.999, n_critic=5, clean_data=True, resume=False,
              ckpt_path='models_save/Gan_QP/checkpoint_20news_clean_20/ckpt_best_1000.pth'):
        self.generator.train()
        self.encoder.train()
        self.discriminator.train()
        if clean_data:
            self.id2token = train_data.dictionary_id2token
            data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        else:
            self.id2token = {v: k for k, v in train_data.dictionary.token2id.items()}
            data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                     collate_fn=train_data.collate_fn)
        optim_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(beta_1, beta_2))
        optim_e_g = torch.optim.Adam([{'params': self.encoder.parameters()}, {'params': self.generator.parameters()}],
                                     lr=lr, betas=(beta_1, beta_2))
        optim_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta_1, beta_2))
        loss_E, loss_G, loss_D = 0, 0, 0
        start_epoch = -1
        if resume:
            checkpoint = torch.load(ckpt_path)
            self.discriminator.load_state_dict(checkpoint['discriminator'])
            self.generator.load_state_dict(checkpoint['generator'])
            self.encoder.load_state_dict(checkpoint['encoder'])
            optim_d.load_state_dict(checkpoint['optimizer_d'])
            optim_e_g.load_state_dict(checkpoint['optimizer_e_d'])
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
                # --------------------
                # train  discriminator
                # --------------------
                optim_d.zero_grad()
                # normalize weight
                bows_real /= torch.sum(bows_real, dim=1, keepdim=True)
                # sample
                topic_fake = torch.from_numpy(np.random.dirichlet(alpha=1.0 * np.ones(self.n_topic) / self.n_topic,
                                                                  size=len(bows_real))).float().to(self.device)
                topic_real = self.encoder(bows_real).detach()
                real_p = torch.cat([topic_real, bows_real], dim=1)
                bows_fake = self.generator(topic_fake).detach()
                fake_p = torch.cat([topic_fake, bows_fake], dim=1)
                loss_D = -self.discriminator(real_p) + self.discriminator(fake_p)
                loss_D = loss_D[:, 0]
                d_normal = 10 * torch.mean(torch.abs(bows_real - bows_fake), dim=1) + torch.mean(
                    torch.abs(topic_real - topic_fake))
                loss_D = torch.mean(-loss_D + 0.5 * loss_D ** 2 / d_normal)
                loss_D.backward()
                optim_d.step()
                if iter_num % n_critic == 0:
                    # train generator and encoder
                    optim_e_g.zero_grad()
                    bows_fake = self.generator(topic_fake).detach()
                    topic_real = self.encoder(bows_real).detach()
                    topic_fake_ = self.encoder(bows_fake)
                    bows_real_ = self.generator(topic_real)

                    p_fake = torch.cat([topic_fake, bows_fake], dim=1)
                    score_fake = self.discriminator(p_fake)
                    p_real = torch.cat([topic_real, bows_real], dim=1)
                    score_real = self.discriminator(p_real)

                    loss_G = torch.mean(score_real - score_fake) + 4 * torch.mean(
                        torch.square(topic_fake - topic_fake_)) + 6 * torch.mean(torch.square(bows_real - bows_real_))
                    loss_G.backward()
                    optim_e_g.step()
                    print(
                        f'Epoch {(epoch + 1):>3d}\tIter {(iter_num + 1):>4d}\tLoss_D:{loss_D.item():<.7f}\tLoss_G:{loss_G.item():<.7f}\t')
            # tensorboardX
            if epoch % 50 == 0:
                self.writer.add_scalars("Train/Loss", {"D_loss": loss_D, "G_loss": loss_G},
                                        epoch)
            if epoch % 250 == 0:
                if test_data is not None:
                    c_a, c_p, npmi = self.get_topic_coherence()
                    if self.max_npmi_value < npmi:
                        self.max_npmi_value = npmi
                        self.max_npmi_step = epoch
                        self.save_model_ckpt(optim_g=optim_g, optim_e=optim_e_g, optim_d=optim_d, epoch=epoch,
                                             max_step=self.max_npmi_step, max_value=self.max_npmi_value)
                    print(c_a, c_p, npmi)
                    print("max_epoch:" + str(self.max_npmi_step) + "  max_value:" + str(self.max_npmi_value))
                    self.writer.add_scalars("Topic Coherence", {'c_a': c_a, 'c_p': c_p, 'npmi': npmi},
                                            epoch)
