"""Test utility functions and session-scoped fixtures."""

import pytest
import torch
import anndata
from ..utils import generate_simulated_dataset


USE_CUDA = torch.cuda.is_available()


@pytest.fixture(scope='session')
def simulated_dataset() -> anndata.AnnData:
    """Generate a small simulated dataset once and make it visible to all tests"""
    return generate_simulated_dataset()
