# -*- coding: utf-8 -*-
# @Time    : 2021/7/7 下午4:50
# @Author  : WuDiDaBinGe
# @FileName: catm_model.py
# @Software: PyCharm
import os
import torch
from torch import nn
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
import time
from model.atm_model import BNTM
from model.cgan import ContrastiveDiscriminator, ContraGenerator
from utils.caculate_coherence import get_coherence_by_local_jar
from utils.contrastive_loss import InstanceLoss, ClusterLoss, Conditional_Contrastive_loss
from utils.tf_idf_data_argument import batch_argument_del_tfidf


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

    def train_with_contra(self, train_data, gamma_temperature, gamma_cluster_temperature, batch_size=64, clip=0.01,
                          lr=1e-4, test_data=None, epochs=100, beta_1=0.5,
                          beta_2=0.999, n_critic=5, clean_data=False, resume=False, ckpt_path=None):
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
            self.writer = SummaryWriter('log/c_atm/20news_clean_2021-07-12-22-03_topic20')
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
                    argument_data_1 = argument_data_1.float().to(self.device)
                    argument_data_2 = argument_data_2.float().to(self.device)
                    # normalize tf-idf weight
                    argument_data_1 /= torch.sum(argument_data_1, dim=1, keepdim=True)
                    argument_data_2 /= torch.sum(argument_data_2, dim=1, keepdim=True)
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
                    # normalize is must be needed in contrastive learning
                    z_i = F.normalize(self.instance_head(h_i), dim=1)
                    z_j = F.normalize(self.instance_head(h_j), dim=1)
                    print(z_j.shape, z_i.shape)
                    instance_loss = contrastive_loss_f(z_i, z_j)
                    loss_e = torch.mean(self.discriminator(p_real))

                    loss_E = loss_e + 0.1 * instance_loss + cluster_loss
                    loss_E.backward()
                    optim_e.step()
                    print(
                        f'Epoch {(epoch + 1):>3d}\tIter {(iter_num + 1):>4d}\tLoss_D:{loss_D.item():<.7f}\tLoss_G:{loss_G.item():<.7f}\tloss_E:{loss_e.item():<.7f}\tcluster_loss:{cluster_loss}\tinstance_loss:{instance_loss.item():<.7f}')

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


class CBTM(BNTM):
    """
    Add contrastive loss in discriminator
    """

    def __init__(self, n_topic, bow_dim, hid_dim, device, task_name):
        super(CBTM, self).__init__(n_topic, bow_dim, hid_dim, device, task_name)
        self.date = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        self.discriminator = ContrastiveDiscriminator(n_topic=n_topic, v_dim=bow_dim, hid_features_dim=int(hid_dim / 2))
        if device is not None:
            self.discriminator = self.discriminator.to(device)

    def train_with_contra(self, train_data, gamma_temperature, batch_size=512, clip=0.01,
                          lr=8e-4, test_data=None, epochs=100, beta_1=0.5,
                          beta_2=0.999, n_critic=5, clean_data=False, resume=False, ckpt_path=None):
        self.generator.train()
        self.encoder.train()
        self.discriminator.train()
        # instance-loss
        contrastive_loss_function = InstanceLoss(batch_size, gamma_temperature, self.device)
        start_epoch = -1
        if clean_data:
            self.id2token = train_data.dictionary_id2token
            data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        else:
            self.id2token = {v: k for k, v in train_data.dictionary.token2id.items()}
            data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                     collate_fn=train_data.collate_fn, drop_last=True)
        optim_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(beta_1, beta_2))
        optim_e = torch.optim.Adam(self.encoder.parameters(), lr=lr, betas=(beta_1, beta_2))
        optim_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr / 100, betas=(beta_1, beta_2))
        loss_E, loss_G, loss_d = 0, 0, 0
        contrastive_loss_real, contrastive_loss_fake = 0, 0
        loss_total_d = 0
        self.writer = SummaryWriter(
            f'log/c_atm_discriminator/{self.task_name}_{self.date}_topic{self.n_topic}')
        if resume and ckpt_path is not None:
            # TODO: 断点续训的时候需要改路径
            self.writer = SummaryWriter('log/c_atm_discriminator/20news_clean_2021-07-14-19-13_topic20')
            checkpoint = torch.load(ckpt_path)
            self.discriminator.load_state_dict(checkpoint['discriminator'])
            self.generator.load_state_dict(checkpoint['generator'])
            self.encoder.load_state_dict(checkpoint['encoder'])
            optim_g.load_state_dict(checkpoint['optimizer_g'])
            optim_d.load_state_dict(checkpoint['optimizer_d'])
            optim_e.load_state_dict(checkpoint['optimizer_e'])
            start_epoch = checkpoint['epoch']

        for epoch in range(start_epoch + 1, epochs + 1):
            for iter_num, data in enumerate(data_loader):
                if clean_data:
                    bows_real, argument_data_1, argument_data_2 = data
                    bows_real = bows_real.float()
                    argument_data_1 = argument_data_1.float().to(self.device)
                    argument_data_2 = argument_data_2.float().to(self.device)
                    # normalize tf-idf weight
                    argument_data_1 /= torch.sum(argument_data_1, dim=1, keepdim=True)
                    argument_data_2 /= torch.sum(argument_data_2, dim=1, keepdim=True)
                else:
                    texts, bows_real = data
                bows_real = bows_real.to(self.device)
                # train  discriminator
                optim_d.zero_grad()
                # normalize tf-idf weight
                bows_real /= torch.sum(bows_real, dim=1, keepdim=True)
                topic_real = self.encoder(bows_real).detach()
                # sample
                topic_fake = torch.from_numpy(np.random.dirichlet(alpha=1.0 * np.ones(self.n_topic) / self.n_topic,
                                                                  size=len(bows_real))).float().to(self.device)
                bows_fake = self.generator(topic_fake).detach()

                # use detach() to stop the gradient backward p(frozen the g and e parameters)
                # advertise loss
                real_score, _ = self.discriminator(topic_real, bows_real)
                fake_score, _ = self.discriminator(topic_fake, bows_fake)
                loss_d = -torch.mean(real_score) + torch.mean(fake_score)
                # contrastive loss
                _, z_i_real = self.discriminator(topic_real, argument_data_1)
                _, z_j_real = self.discriminator(topic_real, argument_data_2)
                contrastive_loss_real = contrastive_loss_function(F.normalize(z_i_real, dim=1),
                                                                  F.normalize(z_j_real, dim=1))
                # get bows_fake argument
                # fake_data_argument_1 = batch_argument_del_tfidf(bows_fake, 0.3).to(self.device)
                # fake_data_argument_2 = batch_argument_del_tfidf(bows_fake, 0.3).to(self.device)
                #
                # _, z_i_fake = self.discriminator(topic_fake, fake_data_argument_1)
                # _, z_j_fake = self.discriminator(topic_fake, fake_data_argument_2)
                # contrastive_loss_fake = contrastive_loss_function(F.normalize(z_i_fake, dim=1),
                #                                                   F.normalize(z_j_fake, dim=1))
                loss_total_d = (contrastive_loss_fake + contrastive_loss_real) + loss_d
                loss_total_d.backward()
                optim_d.step()
                # Clip weights of discriminator
                for p in self.discriminator.parameters():
                    p.data.clamp_(-clip, clip)
                if iter_num % n_critic == 0:
                    # train generator and encoder
                    optim_g.zero_grad()
                    score, _ = self.discriminator(topic_fake, self.generator(topic_fake))
                    loss_G = -torch.mean(score)
                    loss_G.backward()
                    optim_g.step()

                    optim_e.zero_grad()
                    score, _ = self.discriminator(self.encoder(bows_real), bows_real)
                    loss_E = torch.mean(score)
                    loss_E.backward()
                    optim_e.step()
                    print(
                        f'Epoch {(epoch + 1):>3d}\tIter {(iter_num + 1):>4d}\tLoss_D:{loss_d.item():<.7f}\tLoss_G:{loss_G.item():<.7f}\tloss_E:{loss_E.item():<.7f}\tcontrastive_loss_real:{contrastive_loss_real:<.7f}\tcontrastive_loss_fake:{contrastive_loss_fake:<.7f}')

            # tensorboardX
            if epoch % 50 == 0:
                self.writer.add_scalars("Train/Loss", {"D_loss": loss_d, "E_loss": loss_E, "G_loss": loss_G}, epoch)
                self.writer.add_scalars("Train/Contrastive-Loss",
                                        {"instance_loss_real": contrastive_loss_real,
                                         "instance_loss_fake": contrastive_loss_fake}, epoch)
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
                              'discriminator': self.discriminator.state_dict(),
                              'optimizer_g': optim_g.state_dict(),
                              'optimizer_e': optim_e.state_dict(),
                              'optimizer_d': optim_d.state_dict(),
                              'epoch': epoch}
                check_dir = f"models_save/c_atm_discriminator/checkpoint_{self.date}_{self.task_name}_{self.n_topic}"
                if not os.path.isdir(check_dir):
                    os.mkdir(check_dir)
                torch.save(checkpoint, os.path.join(check_dir, f"ckpt_best_{epoch}.pth"))


class GCATM(BNTM):
    """
       Add contrastive loss in Generator
    """

    def __init__(self, n_topic, bow_dim, hid_dim, device, task_name):
        super(GCATM, self).__init__(n_topic, bow_dim, hid_dim, device, task_name)
        self.date = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        self.logdir_name = "gc_atm"
        self.generator = ContraGenerator(n_topic=n_topic, hid_features_dim=hid_dim, v_dim=bow_dim)
        if device is not None:
            self.generator = self.generator.to(device)

    def train_with_contra(self, train_data, gamma_temperature, batch_size=64, clip=0.01,
                          lr=1e-4, test_data=None, epochs=100, beta_1=0.5,
                          beta_2=0.999, n_critic=5, clean_data=False, resume=False, ckpt_path=None):
        self.generator.train()
        self.encoder.train()
        self.discriminator.train()
        # instance-loss
        # contrastive_loss_function = InstanceLoss(batch_size, gamma_temperature, self.device)
        contrastive_loss_function = Conditional_Contrastive_loss(device=self.device, batch_size=batch_size,
                                                                 pos_collected_numerator=True)
        start_epoch = -1
        if clean_data:
            self.id2token = train_data.dictionary_id2token
            data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        else:
            self.id2token = {v: k for k, v in train_data.dictionary.token2id.items()}
            data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                     collate_fn=train_data.collate_fn, drop_last=True)
        optim_g = torch.optim.Adam(self.generator.parameters(), lr=lr / 10, betas=(beta_1, beta_2))
        optim_e = torch.optim.Adam(self.encoder.parameters(), lr=lr, betas=(beta_1, beta_2))
        optim_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta_1, beta_2))
        loss_E, loss_G, loss_d = 0, 0, 0
        contrastive_loss = 0
        loss_total_g = 0
        if not resume:
            self.writer = SummaryWriter(
                f'log/{self.logdir_name}/{self.task_name}_{self.date}_topic{self.n_topic}')
        if resume and ckpt_path is not None:
            # TODO: 断点续训的时候需要改路径
            self.writer = SummaryWriter(f'log/{self.logdir_name}/20news_clean_2021-08-03-16-52_topic100')
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
                # normalize tf-idf weight
                bows_real /= torch.sum(bows_real, dim=1, keepdim=True)
                topic_real = self.encoder(bows_real).detach()
                # sample
                topic_fake = torch.from_numpy(np.random.dirichlet(alpha=1.0 * np.ones(self.n_topic) / self.n_topic,
                                                                  size=len(bows_real))).float().to(self.device)
                bows_fake = self.generator(topic_fake)[0].detach()

                # use detach() to stop the gradient backward p(frozen the g and e parameters)
                # advertise loss
                real_p = torch.cat([topic_real, bows_real], dim=1)
                fake_p = torch.cat([topic_fake, bows_fake], dim=1)
                loss_d = -torch.mean(self.discriminator(real_p)) + torch.mean(self.discriminator(fake_p))

                # contrastive_loss_real = contrastive_loss_function(F.normalize(z_i_real, dim=1),
                #                                                   F.normalize(z_j_real, dim=1))

                loss_d.backward()
                optim_d.step()
                # Clip weights of discriminator
                for p in self.discriminator.parameters():
                    p.data.clamp_(-clip, clip)
                if iter_num % n_critic == 0:
                    # train generator and encoder
                    optim_g.zero_grad()
                    fake_bow, topic_embedding, z_features, labels = self.generator(topic_fake)
                    score = self.discriminator(torch.cat([topic_fake, fake_bow], dim=1))
                    loss_G = -torch.mean(score)
                    fake_cls_mask = self.make_mask(labels, self.n_topic, mask_negatives=True)
                    contrastive_loss = contrastive_loss_function(F.normalize(z_features, dim=1),
                                                                 F.normalize(topic_embedding, dim=1),
                                                                 fake_cls_mask, labels, gamma_temperature
                                                                 )
                    loss_total_g = loss_G + contrastive_loss
                    loss_total_g.backward()
                    optim_g.step()

                    optim_e.zero_grad()
                    score = self.discriminator(torch.cat([self.encoder(bows_real), bows_real], dim=1))
                    loss_E = torch.mean(score)
                    loss_E.backward()
                    optim_e.step()
                    print(
                        f'Epoch {(epoch + 1):>3d}\tIter {(iter_num + 1):>4d}\tLoss_D:{loss_d.item():<.7f}\tLoss_G:{loss_G.item():<.7f}\tloss_E:{loss_E.item():<.7f}\tcontrastive_loss_real:{contrastive_loss:<.7f}')

            # tensorboardX
            if epoch % 50 == 0:
                self.writer.add_scalars("Train/Loss", {"D_loss": loss_d, "E_loss": loss_E, "G_loss": loss_G}, epoch)
                self.writer.add_scalar("Train/Contrastive-Loss", contrastive_loss, epoch)
            if epoch % 250 == 0:
                if test_data is not None:
                    c_a, c_p, npmi = self.get_topic_coherence()
                    if self.max_npmi_value < npmi:
                        self.max_npmi_value = npmi
                        self.max_npmi_step = epoch
                        # save checkpoints only get new max npmi to save
                        self.save_model_ckpt(optim_g=optim_g, optim_e=optim_e, optim_d=optim_d, epoch=epoch,
                                             max_step=self.max_npmi_step, max_value=self.max_npmi_value)
                    print(c_a, c_p, npmi)
                    print("max_epoch:" + str(self.max_npmi_step) + "  max_value:" + str(self.max_npmi_value))
                    self.writer.add_scalars("Topic Coherence", {'c_a': c_a, 'c_p': c_p, 'npmi': npmi},
                                            epoch)

    # 使用local npmi
    def get_topic_coherence(self):
        topic_words = self.show_topic_words()
        print('\n'.join([str(lst) for lst in topic_words]))
        return get_coherence_by_local_jar(topic_words)

    def make_mask(self, labels, n_cls, mask_negatives):
        labels = labels.detach().cpu().numpy()
        n_samples = labels.shape[0]
        if mask_negatives:
            mask_multi, target = np.zeros([n_cls, n_samples]), 1.0
        else:
            mask_multi, target = np.ones([n_cls, n_samples]), 0.0

        for c in range(n_cls):
            c_indices = np.where(labels == c)
            mask_multi[c, c_indices] = target

        return torch.tensor(mask_multi).type(torch.long).to(self.device)
