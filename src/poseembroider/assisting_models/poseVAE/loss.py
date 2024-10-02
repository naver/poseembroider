import torch
import numpy as np
import torch.nn.functional as F


def laplacian_nll(x_tilde, x, log_sigma):
    """ Negative log likelihood of an isotropic Laplacian density """
    log_norm = - (np.log(2) + log_sigma)
    log_energy = - (torch.abs(x_tilde - x)) / torch.exp(log_sigma)
    return - (log_norm + log_energy)


def gaussian_nll(x_tilde, x, log_sigma):
    """ Negative log-likelihood of an isotropic Gaussian density """
    log_norm = - 0.5 * (np.log(2 * np.pi) + log_sigma)
    log_energy = - 0.5 * F.mse_loss(x_tilde, x, reduction='none') / torch.exp(log_sigma)
    return - (log_norm + log_energy)