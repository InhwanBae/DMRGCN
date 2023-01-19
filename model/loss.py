import numpy as np
import torch


def generate_statistics_matrices(V):
    r"""generate mean and covariance matrices from the network output."""

    mu = V[:, :, 0:2]
    sx = V[:, :, 2].exp()
    sy = V[:, :, 3].exp()
    corr = V[:, :, 4].tanh()

    cov = torch.zeros(V.size(0), V.size(1), 2, 2).cuda()
    cov[:, :, 0, 0] = sx * sx
    cov[:, :, 0, 1] = corr * sx * sy
    cov[:, :, 1, 0] = corr * sx * sy
    cov[:, :, 1, 1] = sy * sy

    return mu, cov


def multivariate_loss(V_pred, V_trgt, training=False):
    r"""Batch multivariate loss"""

    mu = V_trgt[:, :, :, 0:2] - V_pred[:, :, :, 0:2]
    mu = mu.unsqueeze(dim=-1)

    sx = V_pred[:, :, :, 2].exp()
    sy = V_pred[:, :, :, 3].exp()
    corr = V_pred[:, :, :, 4].tanh()

    cov = torch.zeros(V_pred.size(0), V_pred.size(1), V_pred.size(2), 2, 2).cuda()

    cov[:, :, :, 0, 0] = sx * sx
    cov[:, :, :, 0, 1] = corr * sx * sy
    cov[:, :, :, 1, 0] = corr * sx * sy
    cov[:, :, :, 1, 1] = sy * sy
    #cov = cov.clamp(min=-1e5, max=1e5)

    pdf = torch.exp(-0.5 * mu.transpose(-2, -1) @ cov.inverse() @ mu)
    pdf = pdf.squeeze() / torch.sqrt(((2 * np.pi) ** 2) * cov.det())

    if training:
        pdf[torch.isinf(pdf) | torch.isnan(pdf)] = 0

    epsilon = 1e-20
    loss = -pdf.clamp(min=epsilon).log()

    return loss.mean()
