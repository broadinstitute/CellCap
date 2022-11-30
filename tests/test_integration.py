"""Integration test ensuring all the expected function calls run without error"""

#  https://github.com/scverse/scvi-tools-skeleton/blob/main/tests/test_skeleton.py

from scvi.data import synthetic_iid

from mypackage import MyModel, MyPyroModel


def test_mymodel():
    n_latent = 5
    adata = synthetic_iid()
    MyModel.setup_anndata(adata, batch_key="batch", labels_key="labels")
    model = MyModel(adata, n_latent=n_latent)
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    model.get_elbo()
    model.get_latent_representation()
    model.get_marginal_ll(n_mc_samples=5)
    model.get_reconstruction_error()
    model.history

    # tests __repr__
    print(model)
