"""Test utility functions and session-scoped fixtures."""

import pytest
import torch
import anndata
import numpy as np

from scvi.data import synthetic_iid


USE_CUDA = torch.cuda.is_available()


def _random_one_hot(n_classes: int, n_samples: int):
    # https://stackoverflow.com/questions/45093615/random-one-hot-matrix-in-numpy
    return np.eye(n_classes)[np.random.choice(n_classes, n_samples)]


@pytest.fixture(scope='session')
def simulated_dataset() -> anndata.AnnData:
    """Generate a small simulated dataset once and make it visible to all tests"""

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
