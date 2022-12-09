"""Integration test ensuring all the expected function calls run without error"""

# https://github.com/scverse/scvi-tools-skeleton/blob/main/tests/test_skeleton.py
# with updates to use pytest fixtures

import pyro

from ..module_pytorch import TorchLatentSpaceAttention
from ..module_pyro import PyroLatentSpaceAttention

from .conftest import simulated_dataset


def test_mymodel(simulated_dataset):
    n_latent = 5
    adata = simulated_dataset
    # TODO modify method calls and setup_anndata call
    TorchLatentSpaceAttention.setup_anndata(adata, batch_key="batch", labels_key="labels")
    model = TorchLatentSpaceAttention(adata, n_latent=n_latent)
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    model.get_elbo()
    model.get_latent_representation()
    model.get_marginal_ll(n_mc_samples=5)
    model.get_reconstruction_error()
    model.history

    # tests __repr__
    print(model)


def test_mypyromodel(simulated_dataset):
    adata = simulated_dataset
    pyro.clear_param_store()
    # TODO modify method calls and setup_anndata call
    PyroLatentSpaceAttention.setup_anndata(adata, batch_key="batch", labels_key="labels")
    model = PyroLatentSpaceAttention(adata)
    model.train(max_epochs=1, train_size=1)
    model.get_latent(adata)
    model.history

    # tests __repr__
    print(model)
