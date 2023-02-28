"""Integration test ensuring all the expected function calls run without error"""

# https://github.com/scverse/scvi-tools-skeleton/blob/main/tests/test_skeleton.py
# with updates to use pytest fixtures

import pytest
import torch
from ..scvi_module import CellCap


USE_CUDA = torch.cuda.is_available()


@pytest.mark.parametrize(
    "cuda",
    [
        False,
        pytest.param(
            True, marks=pytest.mark.skipif(not USE_CUDA, reason="requires CUDA")
        ),
    ],
    ids=lambda b: "cuda" if b else "cpu",
)
@pytest.mark.parametrize("n_latent", [4])
@pytest.mark.parametrize("n_epochs", [2])
def test_basic_functionality(small_real_dataset, n_latent, n_epochs, cuda):
    adata, setup_adata_kwargs, adata_kwargs = small_real_dataset
    print(adata)
    basic_run(
        adata=adata,
        setup_adata_kwargs=setup_adata_kwargs,
        adata_kwargs=adata_kwargs,
        n_latent=n_latent,
        n_epochs=n_epochs,
        cuda=cuda,
        verbose=True,
    )


def basic_run(
    adata, setup_adata_kwargs, adata_kwargs, n_latent, n_epochs, cuda, verbose=False
):
    """Basic run through of user-facing functionality"""

    CellCap.setup_anndata(
        adata,
        **setup_adata_kwargs,
    )
    if verbose:
        print(adata)
    model = CellCap(
        adata=adata,
        n_latent=n_latent,
        **adata_kwargs,
    )
    model.train(
        max_epochs=n_epochs, check_val_every_n_epoch=1, train_size=0.5, use_gpu=cuda
    )
    if verbose:
        print(model.history)

    # tests __repr__
    if verbose:
        print(model)

    return model
