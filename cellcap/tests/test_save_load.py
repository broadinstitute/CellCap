"""Integration test ensuring all the expected function calls run without error"""

import pytest
import torch
from .test_integration import basic_run, setup_model

import tempfile
import os


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
def test_save_and_load(small_real_dataset, n_latent, n_epochs, cuda):
    adata, setup_adata_kwargs, adata_kwargs = small_real_dataset
    print(adata)
    model = basic_run(
        adata=adata,
        setup_adata_kwargs=setup_adata_kwargs,
        adata_kwargs=adata_kwargs,
        n_latent=n_latent,
        n_epochs=n_epochs,
        cuda=cuda,
        verbose=False,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        dirname = os.path.join(tmpdir, "saved_models")

        # save the model
        model.save(dir_path=dirname)

        # load the model
        blank_model = setup_model(
            adata=adata,
            setup_adata_kwargs=setup_adata_kwargs,
            adata_kwargs=adata_kwargs,
            n_latent=n_latent,
            verbose=False,
        )
        loaded_model = blank_model.load(dir_path=dirname, adata=adata, use_gpu=cuda)

    model_state_dict = model.module.state_dict()
    loaded_model_state_dict = loaded_model.module.state_dict()

    for key in model_state_dict.keys():
        print("======================================================")
        print(key)
        print(model_state_dict[key])
        print(loaded_model_state_dict[key])
        torch.testing.assert_close(
            model_state_dict[key],
            loaded_model_state_dict[key],
            msg=f"Parameter '{key}' of loaded model does not agree",
        )
