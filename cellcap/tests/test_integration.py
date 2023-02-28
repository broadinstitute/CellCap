"""Integration test ensuring all the expected function calls run without error"""

# https://github.com/scverse/scvi-tools-skeleton/blob/main/tests/test_skeleton.py
# with updates to use pytest fixtures

from ..module import CellCap


def test_mymodel(small_real_dataset):
    n_latent = 4
    adata = small_real_dataset
    print(adata)
    CellCap.setup_anndata(
        adata,
        pert_key="Condition",
        layer="counts",
        cond_key="X_drug",
        cont_key="X_cont",
        target_key="X_target",
        donor_key="X_donor",
    )
    print(adata)
    model = CellCap(
        adata=adata, n_latent=n_latent, n_drug=12, n_control=2, n_target=14, n_donor=33
    )
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    print(model.history)

    # tests __repr__
    print(model)
