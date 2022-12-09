"""Test utility functions and session-scoped fixtures."""

import pytest
import torch
import anndata

from scvi.data import synthetic_iid


USE_CUDA = torch.cuda.is_available()


@pytest.fixture(scope='session')
def simulated_dataset() -> anndata.AnnData:
    """Generate a small simulated dataset once and make it visible to all tests"""
    return synthetic_iid()
