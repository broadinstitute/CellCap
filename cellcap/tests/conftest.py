"""Test utility functions and session-scoped fixtures."""

import pytest
import torch
import anndata
import os
from ..utils import generate_simulated_dataset


filebase = os.path.dirname(__file__)
test_data_path = os.path.join(filebase, "test.h5ad")
USE_CUDA = torch.cuda.is_available()


@pytest.fixture(scope="session")
def simulated_dataset() -> anndata.AnnData:
    """Generate a small simulated dataset once and make it visible to all tests"""
    return generate_simulated_dataset()


@pytest.fixture(scope="session")
def small_real_dataset() -> anndata.AnnData:
    adata = anndata.read_h5ad(test_data_path)
    return adata
