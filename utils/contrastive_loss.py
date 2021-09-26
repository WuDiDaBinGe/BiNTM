# -*- coding: utf-8 -*-
# @Time    : 2021/7/8 上午9:49
# @Author  : WuDiDaBinGe
# @FileName: contrastive_loss.py
# @Software: PyCharm
import torch
import torch.nn as nn
import math
import numpy as np


class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)
        # sim (N * N)
        sim = torch.matmul(z, z.T) / self.temperature
        # sim_i_j (batch_size)
        sim_i_j = torch.diag(sim, self.batch_size)
        # sim_j_i (batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        # positive (N)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        # labels  N
        labels = torch.zeros(N).to(positive_samples.device).long()
        # logits (N,N-1)
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class ClusterLoss(nn.Module):
    def __init__(self, class_num, device, temperature):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss


class Conditional_Contrastive_loss(torch.nn.Module):
    def __init__(self, device, batch_size, pos_collected_numerator):
        super(Conditional_Contrastive_loss, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.pos_collected_numerator = pos_collected_numerator
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def remove_diag(self, M):
        h, w = M.shape
        assert h == w, "h and w should be same"
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask)
        mask = (mask).type(torch.bool).to(self.device)
        return M[mask].view(h, -1)

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, inst_embed, proxy, negative_mask, labels, temperature, margin=0):
        # batch_size * batch_size
        similarity_matrix = self.calculate_similarity_matrix(inst_embed, inst_embed)
        instance_zone = torch.exp((self.remove_diag(similarity_matrix) - margin) / temperature)
        # 计算每个instance 与他的标签embedding的相似度
        inst2proxy_positive = torch.exp((self.cosine_similarity(inst_embed, proxy) - margin) / temperature)

        if self.pos_collected_numerator:
            mask_4_remove_negatives = negative_mask[labels]
            mask_4_remove_negatives = self.remove_diag(mask_4_remove_negatives)
            inst2inst_positives = instance_zone * mask_4_remove_negatives

            numerator = inst2proxy_positive + inst2inst_positives.sum(dim=1)
        else:
            numerator = inst2proxy_positive

        denomerator = torch.cat([torch.unsqueeze(inst2proxy_positive, dim=1), instance_zone], dim=1).sum(dim=1)
        # 乘 这个temperature的意义是什么
        criterion = -torch.log(temperature * (numerator / denomerator)).mean()
        return criterion


def make_mask(labels, n_cls, mask_negatives, device):
    labels = labels.detach().cpu().numpy()
    n_samples = labels.shape[0]
    if mask_negatives:
        mask_multi, target = np.zeros([n_cls, n_samples]), 1.0
    else:
        mask_multi, target = np.ones([n_cls, n_samples]), 0.0

    for c in range(n_cls):
        c_indices = np.where(labels == c)
        mask_multi[c, c_indices] = target

    return torch.tensor(mask_multi).type(torch.long)


if __name__ == '__main__':
    device = torch.device('cuda')
    a = torch.rand((64, 20))
    b = torch.rand((64, 20))
    labels = torch.randint(low=0, high=5, size=(64,))
    negetive_mask = make_mask(labels, 5, True, device)
    criten_instance = Conditional_Contrastive_loss(device, 64, pos_collected_numerator=True)
    print(criten_instance(a, b, negetive_mask, labels, 0.1))
