"""Integration test ensuring all the expected function calls run without error"""

# https://github.com/scverse/scvi-tools-skeleton/blob/main/tests/test_skeleton.py
# with updates to use pytest fixtures

import scanpy as sc
from ..module import CellCap
#from .conftest import simulated_dataset
data = '../tests/test.h5ad'

def test_mymodel(data):
    n_latent = 4
    adata = sc.read_h5ad(data)
    print(adata)
    kwargs = {}
    for k in adata.obsm_keys():
        kwargs.update({f"{k}_key": k})
    CellCap.setup_anndata(adata,
                          labels_key='control',
                          pert_key='Condition',
                          layer="counts",
                          cond_key='X_drug',
                          cont_key='X_cont',
                          target_key='X_target',
                          donor_key='X_donor',
                          **kwargs)
    print(adata)
    model = CellCap(adata=adata, n_latent=n_latent,
                    n_drug=12,n_control=2,
                    n_target=14,n_donor=33)
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    print(model.history)

    # tests __repr__
    print(model)
