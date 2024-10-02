import torch
import torch.nn.functional as F


def BBC(scores):
    # build the ground truth label tensor: the diagonal corresponds to the
    # correct classification
    GT_labels = torch.arange(scores.shape[0], device=scores.device).long()
    loss = F.cross_entropy(scores, GT_labels) # mean reduction
    return loss


def symBBC(scores):
    x2y_loss = BBC(scores)
    y2x_loss = BBC(scores.t())
    return (x2y_loss + y2x_loss) / 2.0