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
        # 乘 这个temperature的意义是什么(这个是多余的)
        criterion = -torch.log((numerator / denomerator)).mean()
        return criterion

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        # contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_feature = features
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

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

class ConditionalContrastiveLoss(torch.nn.Module):
    def __init__(self, num_classes, temperature, master_rank, DDP):
        super(ConditionalContrastiveLoss, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.master_rank = master_rank
        self.DDP = DDP
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _make_neg_removal_mask(self, labels):
        labels = labels.detach().cpu().numpy()
        n_samples = labels.shape[0]
        mask_multi, target = np.zeros([self.num_classes, n_samples]), 1.0
        for c in range(self.num_classes):
            c_indices = np.where(labels == c)
            mask_multi[c, c_indices] = target
        return torch.tensor(mask_multi).type(torch.long).to(self.master_rank)

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def _remove_diag(self, M):
        h, w = M.shape
        assert h == w, "h and w should be same"
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask)
        mask = (mask).type(torch.bool).to(self.master_rank)
        return M[mask].view(h, -1)

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, embed, proxy, label, **_):
        # if self.DDP:
        #     embed = torch.cat(misc.GatherLayer.apply(embed), dim=0)
        #     proxy = torch.cat(misc.GatherLayer.apply(proxy), dim=0)
        #     label = torch.cat(misc.GatherLayer.apply(label), dim=0)

        sim_matrix = self.calculate_similarity_matrix(embed, embed)
        sim_matrix = torch.exp(self._remove_diag(sim_matrix) / self.temperature)
        neg_removal_mask = self._remove_diag(self._make_neg_removal_mask(label)[label])
        sim_pos_only = neg_removal_mask * sim_matrix

        emb2proxy = torch.exp(self.cosine_similarity(embed, proxy) / self.temperature)

        numerator = emb2proxy + sim_pos_only.sum(dim=1)
        denomerator = torch.cat([torch.unsqueeze(emb2proxy, dim=1), sim_matrix], dim=1).sum(dim=1)
        return -torch.log(numerator / denomerator).mean()

class MiConditionalContrastiveLoss(torch.nn.Module):
    def __init__(self, num_classes, temperature, master_rank, DDP):
        super(MiConditionalContrastiveLoss, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.master_rank = master_rank
        self.DDP = DDP
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _make_neg_removal_mask(self, labels):
        labels = labels.detach().cpu().numpy()
        n_samples = labels.shape[0]
        mask_multi, target = np.zeros([self.num_classes, n_samples]), 1.0
        for c in range(self.num_classes):
            c_indices = np.where(labels == c)
            mask_multi[c, c_indices] = target
        return torch.tensor(mask_multi).type(torch.long).to(self.master_rank)

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def _remove_diag(self, M):
        h, w = M.shape
        assert h == w, "h and w should be same"
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask)
        mask = (mask).type(torch.bool).to(self.master_rank)
        return M[mask].view(h, -1)

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, mi_embed, mi_proxy, label, **_):
        # if self.DDP:
        #     mi_embed = torch.cat(misc.GatherLayer.apply(mi_embed), dim=0)
        #     mi_proxy = torch.cat(misc.GatherLayer.apply(mi_proxy), dim=0)
        #     label = torch.cat(misc.GatherLayer.apply(label), dim=0)

        sim_matrix = self.calculate_similarity_matrix(mi_embed, mi_embed)
        sim_matrix = torch.exp(self._remove_diag(sim_matrix) / self.temperature)
        neg_removal_mask = self._remove_diag(self._make_neg_removal_mask(label)[label])
        sim_pos_only = neg_removal_mask * sim_matrix

        emb2proxy = torch.exp(self.cosine_similarity(mi_embed, mi_proxy) / self.temperature)

        numerator = emb2proxy + sim_pos_only.sum(dim=1)
        denomerator = torch.cat([torch.unsqueeze(emb2proxy, dim=1), sim_matrix], dim=1).sum(dim=1)
        return -torch.log(numerator / denomerator).mean()



if __name__ == '__main__':
    device = torch.device('cuda')
    a = torch.rand((64, 20))
    b = torch.rand((64, 20))
    labels = torch.randint(low=0, high=5, size=(64,))
    negetive_mask = make_mask(labels, 5, True, device)
    criten_instance = Conditional_Contrastive_loss(device, 64, pos_collected_numerator=True)
    print(criten_instance(a, b, negetive_mask, labels, 0.1))
