# -*- coding: utf-8 -*-
# @Time    : 2021/7/30 下午8:19
# @Author  : WuDiDaBinGe
# @FileName: cgan.py
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

class ContraGenerator(nn.Module):
    def __init__(self, n_topic, hid_features_dim, v_dim, c_embedding_z_dim=128):
        super(ContraGenerator, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=n_topic, embedding_dim=c_embedding_z_dim)
        self.generator_encoder = nn.Sequential(
            *block(n_topic, hid_features_dim),
        )
        self.generator_head = nn.Sequential(
            nn.Linear(hid_features_dim, v_dim),
            nn.Softmax(dim=1)
        )
        # self.softmax = nn.Softmax(dim=1)
        # doc instance project for contrastive loss
        self.project_head = nn.Sequential(
            nn.Linear(hid_features_dim, hid_features_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_features_dim, c_embedding_z_dim),
        )

    def forward(self, theta):
        # get max index in topic distribution
        topic_label = torch.argmax(theta, dim=1)
        topic_embedding = self.embedding(topic_label)

        hid_features = self.generator_encoder(theta)
        bow_out = self.generator_head(hid_features)

        z_features = self.project_head(hid_features)
        return bow_out, topic_embedding, z_features, topic_label

    def inference(self, theta):
        hid_features = self.generator_encoder(theta)
        bow_out = self.generator_head(hid_features)
        return bow_out


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
