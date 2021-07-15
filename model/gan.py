# -*- coding: utf-8 -*-
# @Time    : 2021/6/19 下午1:35
# @Author  : WuDiDaBinGe
# @FileName: gan.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F


def block(in_feat, out_feat, normalize=True):
    layers = [nn.Linear(in_feat, out_feat, bias=True)]
    if normalize:
        layers.append(nn.BatchNorm1d(out_feat))
    layers.append(nn.LeakyReLU(inplace=True))
    return layers


class Generator(nn.Module):
    def __init__(self, n_topic, hid_dim, v_dim):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            *block(n_topic, hid_dim),
            nn.Linear(hid_dim, v_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        output = self.generator(inputs)
        return output

    def inference(self, theta):
        return self.generator(theta)


class Encoder(nn.Module):
    def __init__(self, v_dim, hid_dim, n_topic):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            *block(v_dim, hid_dim),
            nn.Linear(hid_dim, n_topic),
            nn.Softmax(dim=1),
        )

    def forward(self, inputs):
        output = self.encoder(inputs)
        return output


class Discriminator(nn.Module):
    def __init__(self, n_topic, hid_dim, v_dim):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            *block(n_topic + v_dim, hid_dim),
            nn.Linear(hid_dim, 1),
        )

    def forward(self, reps):
        score = self.discriminator(reps)
        return score


class ContrastiveDiscriminator(nn.Module):
    def __init__(self, n_topic, v_dim, hid_features_dim, z_features_dim=128):
        super(ContrastiveDiscriminator, self).__init__()
        # doc hidden features
        self.discriminator_encoder = nn.Sequential(
            *block(v_dim, 2048),
            *block(2048, 1024),
            *block(1024, hid_features_dim),
        )
        # doc instance project for contrastive loss
        self.project_head = nn.Sequential(
            nn.Linear(hid_features_dim, hid_features_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_features_dim, z_features_dim)
        )
        # doc hidden features + topic
        self.score_head = nn.Sequential(
            *block(n_topic + hid_features_dim, 256),
            nn.Linear(256, 1)
        )

    def forward(self, topic_distribute, doc_bow):
        doc_hidden_features = self.discriminator_encoder(doc_bow)
        contrastive_features = self.project_head(doc_hidden_features)
        p_join = torch.cat([topic_distribute, doc_hidden_features], dim=1)
        score = self.score_head(p_join)
        return score, contrastive_features
