import torch 
import torch.nn.functional as F
import torch.nn as nn

import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = torch.norm(anchor - positive, p=2, dim=1)
        distance_negative = torch.norm(anchor - negative, p=2, dim=1)

        # Triplet loss: max(0, positive_dist - negative_dist + margin)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = torch.norm(output1 - output2, p=2, dim=1)

        # Contrastive loss: (1 - label) * d^2 + label * max(0, margin - d)^2
        positive_loss = (1 - label) * torch.pow(euclidean_distance, 2)
        negative_loss = label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)

        loss_contrastive = torch.mean(positive_loss + negative_loss)
        return loss_contrastive


class DepthAdjustedLoss(torch.nn.Module):
    def __init__(self, loss=nn.CrossEntropyLoss(reduction='none')):
        super(DepthAdjustedLoss, self).__init__()
        self.loss = loss

    def forward(self, *args, depths=None):
        adapted_loss = self.loss(*args)
        if (depths is not None):
            depths = depths.type(torch.FloatTensor)
            adapted_loss = torch.mul(depths,adapted_loss)
            adapted_loss = torch.mean(adapted_loss)
        return adapted_loss

