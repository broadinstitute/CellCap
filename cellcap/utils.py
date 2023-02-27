"""Utility functions"""

import torch
import torch.nn.functional as F
import anndata
import numpy as np

from scvi.data import synthetic_iid

def entropy(x, temp=1.0):
    p = F.softmax(x / temp, dim=1)# + 1e-8
    logp = F.log_softmax(x / temp, dim=1)# + 1e-8
    return -(p*logp).sum(dim=1)

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def cal_off_diagonal_corr(z):
    c = z[0, :, :].T @ z[0, :, :]
    off_diag = off_diagonal(c).pow_(2).sum()
    for i in range(1,z.shape[0]):
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
    del adata.obsm['protein_expression']

    # add in necessary fields
    obsm_data = {'cond': _random_one_hot(2, n),
                 'cont': _random_one_hot(2, n),
                 'target': _random_one_hot(2, n),
                 'donor': _random_one_hot(2, n)}
    for k, v in obsm_data.items():
        adata.obsm[k] = v
    adata.obs['pert'] = _random_one_hot(2, n)[:, 0]

    return adata

class Metrics:

    def __init__(self):
        self.metrics = {}

    def add(self, k, v):
        self.metrics.update({k: v})

    def remove(self, k):
        self.metrics.pop(k)

    def __iter__(self):
        for k, v in self.metrics.items():
            yield k, v


_METRICS_TO_LOG = Metrics()

def log_metric(name: str, tensor: torch.Tensor) -> None:
    """Log a value, saving it to the model history.
    Make use of logging in pytorch-lightning.

    The idea is to log scalars, not full tensors. This function accepts tensors
    and flattens them, assigning a unique name to each dimension. However, it
    is a bad idea to try to log a tensor with a very large number of elements.

    Parameters
    ----------
    name: Name of metric
    tensor: Value to log
    """
    tensor = tensor.detach()
    flat_tensor = tensor.flatten()
    if len(flat_tensor) > 10:
        raise UserWarning(f"You are logging a tensor with {len(tensor)} "
                          f"elements. Each will have a separate column in "
                          f"the trainer object history, with an underscore "
                          f"appended to denote the entry.")

    if tensor.numel() == 1:
        _METRICS_TO_LOG.add(name, tensor)
    elif tensor.numel() > 1:
        for i, val in enumerate(flat_tensor):
            _METRICS_TO_LOG.add(f"{name}_{i}", val)
