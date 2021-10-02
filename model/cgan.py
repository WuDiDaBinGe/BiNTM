# -*- coding: utf-8 -*-
# @Time    : 2021/7/30 下午8:19
# @Author  : WuDiDaBinGe
# @FileName: cgan.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.utils as spectral_norm
import torch.nn.functional as F


def block(in_feat, out_feat, normalize=True):
    layers = [nn.Linear(in_feat, out_feat, bias=True)]
    if normalize:
        layers.append(nn.BatchNorm1d(out_feat))
    layers.append(nn.LeakyReLU(inplace=True))
    return layers


class Generator_H(nn.Module):
    def __init__(self, n_topic, hid_dim, v_dim):
        super(Generator_H, self).__init__()
        self.generator_h = nn.Sequential(
            *block(n_topic, hid_dim),
        )
        self.generator = nn.Sequential(
            nn.Linear(hid_dim, v_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        output = self.generator_h(inputs)
        return self.generator(output), output

    def inference(self, theta):
        output = self.generator_h(theta)
        return self.generator(output)


class Encoder_H(nn.Module):
    def __init__(self, v_dim, hid_dim, n_topic):
        super(Encoder_H, self).__init__()
        self.encoder_hid = nn.Sequential(
            *block(v_dim, hid_dim)
        )
        self.encoder = nn.Sequential(
            nn.Linear(hid_dim, n_topic),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        output = self.encoder_hid(inputs)
        return self.encoder(output), output


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


class ContraMeanGenerator(nn.Module):
    def __init__(self, n_topic, hid_features_dim, v_dim, c_embedding_z_dim=64):
        super(ContraMeanGenerator, self).__init__()
        self.topic_num = n_topic
        self.embedding = nn.EmbeddingBag(num_embeddings=n_topic, embedding_dim=c_embedding_z_dim, mode='sum')
        self.generator = nn.Sequential(
            *block(n_topic, hid_features_dim),
            nn.Linear(hid_features_dim, v_dim),
            nn.Softmax(dim=1)
        )
        # self.softmax = nn.Softmax(dim=1)
        # doc instance project for contrastive loss
        self.project_head = nn.Sequential(
            nn.Linear(v_dim, c_embedding_z_dim),
        )

    def forward(self, theta, inputs_):
        topic_label = torch.argmax(theta, dim=1)
        topic_embedding = self.embedding(inputs_, per_sample_weights=theta).squeeze(dim=0)
        bow_out = self.generator(theta)

        z_features = self.project_head(bow_out)
        return bow_out, topic_embedding, z_features, topic_label

    def inference(self, theta):
        bow_out = self.generator(theta)
        return bow_out


class WordEmbeddingDiscriminator(nn.Module):
    def __init__(self, n_topic, hid_features_dim, v_dim, pretrain_embedding):
        super(WordEmbeddingDiscriminator, self).__init__()
        self.topic_num = n_topic
        embedding = torch.load(pretrain_embedding)
        self.word_embedding = nn.EmbeddingBag.from_pretrained(embedding)
        self.discriminator = nn.Sequential(
            *block(n_topic + v_dim, hid_features_dim),
            nn.Linear(hid_features_dim, 1),
        )

    def forward(self, theta, bow, word_inputs):
        p_join = torch.cat([theta, bow])
        score = self.discriminator(p_join)
        word_embedding = self.embedding_sum(word_inputs, per_sample_weights=bow).squeeze(dim=0)
        return score, word_embedding


class ContrastiveDiscriminator(nn.Module):
    def __init__(self, n_topic, v_dim, hid_features_dim, z_features_dim=64):
        super(ContrastiveDiscriminator, self).__init__()

        # encoder bow
        self.encoder_bow = nn.Sequential(
            *block(v_dim, hid_features_dim),
        )

        # encoder topic
        self.encoder_topic = nn.Sequential(
            *block(n_topic, z_features_dim),
        )
        # adv head
        self.x_score = nn.Linear(hid_features_dim, 1)
        self.z_score = nn.Linear(z_features_dim, 1)
        self.discriminator = nn.Sequential(
            *block(v_dim + n_topic, hid_features_dim),
            nn.Linear(hid_features_dim, 1)
        )

        # doc hidden features
        self.bow_project_head = nn.Sequential(
            nn.Linear(hid_features_dim, hid_features_dim),
            nn.ReLU(),
            nn.Linear(hid_features_dim, z_features_dim)
        )

        self.label_project_head = nn.Sequential(
            nn.Linear(z_features_dim, hid_features_dim),
            nn.ReLU(),
            nn.Linear(hid_features_dim, z_features_dim)
        )

    def forward(self, topic_distribute, doc_bow):
        doc_h = self.encoder_bow(doc_bow)
        x_score = self.x_score(doc_h)
        topic_h = self.encoder_topic(topic_distribute)
        z_score = self.z_score(topic_h)
        p_join = torch.cat([topic_distribute, doc_bow], dim=1)
        score = self.discriminator(p_join)

        doc_embed = self.bow_project_head(doc_h)
        label_embed = self.label_project_head(topic_h)
        return x_score, z_score, score, doc_embed, label_embed


class BigDiscriminator(nn.Module):
    def __init__(self, n_topic, v_dim, hid_features_dim, z_features_dim=128):
        super(BigDiscriminator, self).__init__()

        # encoder bow
        self.encoder_bow = nn.Sequential(
            *block(v_dim, hid_features_dim),
        )

        # encoder topic
        self.encoder_topic = nn.Sequential(
            *block(n_topic, z_features_dim),
        )
        # adv head
        self.x_score = nn.Linear(hid_features_dim, 1)
        self.z_score = nn.Linear(z_features_dim, 1)
        self.discriminator = nn.Sequential(
            *block(hid_features_dim + z_features_dim, hid_features_dim),
            nn.Linear(hid_features_dim, 1),
        )

    def forward(self, topic_distribute, doc_bow):
        doc_h = self.encoder_bow(doc_bow)
        x_score = self.x_score(doc_h)
        topic_h = self.encoder_topic(topic_distribute)
        z_score = self.z_score(topic_h)
        p_join = torch.cat([topic_h, doc_h], dim=1)
        score = self.discriminator(p_join)

        # 总分数属于 -1，1
        return x_score, z_score, score


if __name__ == '__main__':
    model = ContraMeanGenerator(n_topic=20, v_dim=3000, hid_features_dim=1024)
    for name, param in model.named_parameters():
        print(name)
        if param.requires_grad:
            print(param.grad)
