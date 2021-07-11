# -*- coding: utf-8 -*-
# @Time    : 2021/7/7 下午4:50
# @Author  : WuDiDaBinGe
# @FileName: catm_model.py
# @Software: PyCharm
import os

import torch
from torch import nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
import time
from model.atm_model import BNTM
from utils.contrastive_loss import InstanceLoss, ClusterLoss


class CATM(BNTM):
    def __init__(self, n_topic, bow_dim, hid_dim, device=None, task_name=None):
        super(CATM, self).__init__(n_topic, bow_dim, hid_dim, device, task_name)
        self.date = time.strftime("%Y-%m-%d-%H-%M", time.localtime())

        self.instance_head = nn.Sequential(
            nn.Linear(self.n_topic, self.n_topic),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_topic, 128)
        )

        if device is not None:
            self.instance_head = self.instance_head.to(device)

    def train(self, train_data, batch_size=512, clip=0.01, lr=8e-4, test_data=None, epochs=100, beta_1=0.5,
              beta_2=0.999, n_critic=5, clean_data=False, resume=False, gamma_temperature=0.5,
              gamma_cluster_temperature=1, ckpt_path=None):
        self.generator.train()
        self.encoder.train()
        self.discriminator.train()
        self.instance_head.train()
        # instance-loss
        contrastive_loss_f = InstanceLoss(batch_size, gamma_temperature, self.device)
        # cluster-loss
        cluster_loss_f = ClusterLoss(self.n_topic, temperature=gamma_cluster_temperature, device=self.device)
        start_epoch = -1
        if clean_data:
            self.id2token = train_data.dictionary_id2token
            data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        else:
            self.id2token = {v: k for k, v in train_data.dictionary.token2id.items()}
            data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                     collate_fn=train_data.collate_fn)
        optim_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(beta_1, beta_2))
        optim_e = torch.optim.Adam([{'params': self.encoder.parameters()}, {'params': self.instance_head.parameters()}],
                                   lr=lr, betas=(beta_1, beta_2))
        optim_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta_1, beta_2))
        loss_E, loss_G, loss_D = 0, 0, 0
        loss_e, instance_loss, cluster_loss = 0, 0, 0
        self.writer = SummaryWriter(
            f'log/c_atm/{self.task_name}_{self.date}_topic{self.n_topic}')
        if resume and ckpt_path is not None:
            self.writer = SummaryWriter('log/c_atm/20news_clean_2021-07-09-17-35_topic20')
            checkpoint = torch.load(ckpt_path)
            self.discriminator.load_state_dict(checkpoint['discriminator'])
            self.generator.load_state_dict(checkpoint['generator'])
            self.encoder.load_state_dict(checkpoint['encoder'])
            self.instance_head.load_state_dict(checkpoint['instance_head'])
            optim_g.load_state_dict(checkpoint['optimizer_g'])
            optim_d.load_state_dict(checkpoint['optimizer_d'])
            optim_e.load_state_dict(checkpoint['optimizer_e'])
            start_epoch = checkpoint['epoch']

        for epoch in range(start_epoch + 1, epochs + 1):
            for iter_num, data in enumerate(data_loader):
                if clean_data:
                    bows_real, argument_data_1, argument_data_2 = data
                    bows_real = bows_real.float()
                    argument_data_1 = argument_data_1.float()
                    argument_data_2 = argument_data_2.float()
                    argument_data_1 = argument_data_1.to(self.device)
                    argument_data_2 = argument_data_2.to(self.device)
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
                    # add contrastive loss
                    h_i = self.encoder(argument_data_1)
                    h_j = self.encoder(argument_data_2)
                    z_i = self.instance_head(h_i)
                    z_j = self.instance_head(h_j)

                    cluster_loss = cluster_loss_f(h_i, h_j)
                    instance_loss = contrastive_loss_f(z_i, z_j)
                    loss_e = torch.mean(self.discriminator(p_real))

                    loss_E = loss_e + instance_loss + cluster_loss
                    loss_E.backward()
                    optim_e.step()
                    print(
                        f'Epoch {(epoch + 1):>3d}\tIter {(iter_num + 1):>4d}\tLoss_D:{loss_D.item():<.7f}\tLoss_G:{loss_G.item():<.7f}\tloss_E:{loss_e.item():<.7f}\tcluster_loss:{cluster_loss.item()}\tinstance_loss:{instance_loss}')

            # tensorboardX
            if epoch % 50 == 0:
                self.writer.add_scalars("Train/Loss", {"D_loss": loss_D, "E_loss": loss_e, "G_loss": loss_G}, epoch)
                self.writer.add_scalars("Train/Contrastive-Loss",
                                        {"instance_loss": instance_loss, "cluster_loss": cluster_loss}, epoch)
            if epoch % 250 == 0:
                if test_data is not None:
                    c_a, c_p, npmi = self.get_topic_coherence()
                    print(c_a, c_p, npmi)
                    self.writer.add_scalars("Topic Coherence", {'c_a': c_a, 'c_p': c_p, 'npmi': npmi},
                                            epoch)
            # save checkpoints
            if epoch % 500 == 0:
                checkpoint = {'generator': self.generator.state_dict(),
                              'encoder': self.encoder.state_dict(),
                              'instance_head': self.instance_head.state_dict(),
                              'discriminator': self.discriminator.state_dict(),
                              'optimizer_g': optim_g.state_dict(),
                              'optimizer_e': optim_e.state_dict(),
                              'optimizer_d': optim_d.state_dict(),
                              'epoch': epoch}
                check_dir = f"./models_save/c_atm/checkpoint_{self.date}_{self.task_name}_{self.n_topic}"
                if not os.path.isdir(check_dir):
                    os.mkdir(check_dir)
                torch.save(checkpoint, os.path.join(check_dir, f"ckpt_best_{epoch}.pth"))
