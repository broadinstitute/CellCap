"""Utility functions"""

import numpy as np

import torch
import torch.nn.functional as F

from scvi.data import synthetic_iid


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
    elif classname.find("Linear") != -1:
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


def permute_dims(z):
    assert z.dim() == 2

    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)


def entropy(x, temp=1.0):
    p = F.softmax(x / temp, dim=1)  # + 1e-8
    logp = F.log_softmax(x / temp, dim=1)  # + 1e-8
    return -(p * logp).sum(dim=1)


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def cal_off_diagonal_corr(z):
    c = z[0, :, :].T @ z[0, :, :]
    off_diag = off_diagonal(c).pow_(2).sum()
    for i in range(1, z.shape[0]):
        c = z[i, :, :].T @ z[i, :, :]
        off_diag += off_diagonal(c).pow_(2).sum()
    return off_diag


def _random_one_hot(n_classes: int, n_samples: int):
    # https://stackoverflow.com/questions/45093615/random-one-hot-matrix-in-numpy
    return np.eye(n_classes)[np.random.choice(n_classes, n_samples)]


def generate_simulated_dataset() -> anndata.AnnData:
    # random data via scvi-tools
    adata = synthetic_iid()
    n = adata.shape[0]
    del adata.obsm["protein_expression"]

    # add in necessary fields
    obsm_data = {
        "cond": _random_one_hot(2, n),
        "cont": _random_one_hot(2, n),
        "target": _random_one_hot(2, n),
        "donor": _random_one_hot(2, n),
    }
    for k, v in obsm_data.items():
        adata.obsm[k] = v
    adata.obs["pert"] = _random_one_hot(2, n)[:, 0]

    return adata
