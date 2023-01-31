"""Integration test ensuring all the expected function calls run without error"""

# https://github.com/scverse/scvi-tools-skeleton/blob/main/tests/test_skeleton.py
# with updates to use pytest fixtures

from ..module_pytorch import CellCap
from .conftest import simulated_dataset


def test_mymodel(simulated_dataset):
    n_latent = 4
    adata = simulated_dataset
    print(adata)
    kwargs = {}
    for k in adata.obsm_keys():
        kwargs.update({f"{k}_key": k})
    CellCap.setup_anndata(adata=adata,
                          batch_key="batch",
                          labels_key="labels",
                          pert_key="pert",
                          **kwargs)
    print(adata)
    model = CellCap(adata=adata, n_latent=n_latent)
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    print(model.history)

    # tests __repr__
    print(model)
