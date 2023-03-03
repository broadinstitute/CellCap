"""Test utility functions and session-scoped fixtures."""

import pytest
import torch
import anndata
import os
from ..utils import generate_simulated_dataset
from typing import Dict, Tuple, Any


filebase = os.path.dirname(__file__)
test_data_path = os.path.join(filebase, "test.h5ad")
USE_CUDA = torch.cuda.is_available()


@pytest.fixture(scope="session")
def simulated_dataset() -> anndata.AnnData:
    """Generate a small simulated dataset once and make it visible to all tests"""
    return generate_simulated_dataset()


@pytest.fixture(scope="session")
def small_real_dataset() -> Tuple[anndata.AnnData, Dict[str, Any], Dict[str, Any]]:
    adata = anndata.read_h5ad(test_data_path)

    # fixed aspects of this dataset
    setup_adata_kwargs = {
        "pert_key": "Condition",
        "layer": "counts",
        "cond_key": "X_drug",
        "cont_key": "X_cont",
        "target_key": "X_target",
        "donor_key": "X_donor",
    }
    adata_kwargs = {"n_drug": 12, "n_control": 2, "n_target": 14, "n_donor": 33}

    return adata, setup_adata_kwargs, adata_kwargs
