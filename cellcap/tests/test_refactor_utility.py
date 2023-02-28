"""This is a utility to help with refactoring.
Only to be run as a temporary tool.

Summary:

When refactoring, the idea is to rearrange code without changing the functionality
or the output. This is a utility that helps ensure that your code changes have
not affected the output.

Usage:

1. Set CREATE_NEW_REFACTORING_BENCHMARK = True
2. Run pytest from the root of the repository
    - This will create a "benchmark" and save it in .refactoring_benchmark/
3. Set CREATE_NEW_REFACTORING_BENCHMARK = False
4. Set RUN_REFACTORING_TESTS_AGAINST_BENCHMARK = True
5. Do your code refactoring
    - Run pytest as desired, and if test_refactor fails, then your code changes
      have changed the output
6. When you are done, set RUN_REFACTORING_TESTS_AGAINST_BENCHMARK = False
   so that this file is ignored during normal operation of pytest

"""

import torch
import os
import shutil
from .test_integration import basic_run
from ..scvi_module import CellCap

CREATE_NEW_REFACTORING_BENCHMARK = False
RUN_REFACTORING_TESTS_AGAINST_BENCHMARK = True

USE_CUDA = torch.cuda.is_available()
DIRECTORY = ".refactoring_benchmark/"
BENCHMARK_MODEL_PATH = os.path.join(DIRECTORY, "benchmark_model/")
REFACTORED_MODEL_PATH = os.path.join(DIRECTORY, "refactored_model/")


def test_create_refactoring_benchmark(small_real_dataset):
    if not CREATE_NEW_REFACTORING_BENCHMARK:
        pass
    else:
        if os.path.exists(BENCHMARK_MODEL_PATH):
            shutil.rmtree(BENCHMARK_MODEL_PATH)
        trained_model = train_model(small_real_dataset)
        trained_model.save(BENCHMARK_MODEL_PATH)


def test_compare_against_refactoring_benchmark(small_real_dataset):
    if not RUN_REFACTORING_TESTS_AGAINST_BENCHMARK:
        pass
    else:
        assert os.path.exists(BENCHMARK_MODEL_PATH), "Create a benchmark first, see doc"
        if os.path.exists(REFACTORED_MODEL_PATH):
            shutil.rmtree(REFACTORED_MODEL_PATH)
        trained_model = train_model(small_real_dataset)
        trained_model.save(REFACTORED_MODEL_PATH)
        adata, _, _ = small_real_dataset
        assert models_equal(
            REFACTORED_MODEL_PATH, BENCHMARK_MODEL_PATH, adata
        ), "Refactoring has changed the output"


def train_model(small_real_dataset):
    adata, setup_adata_kwargs, adata_kwargs = small_real_dataset
    print(adata)
    trained_model = basic_run(
        adata=adata,
        setup_adata_kwargs=setup_adata_kwargs,
        adata_kwargs=adata_kwargs,
        n_latent=8,
        n_epochs=10,
        cuda=False,
        verbose=False,
    )
    return trained_model


def models_equal(dir1, dir2, adata) -> bool:
    """Check if two saved models are identical

    https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351/9

    Args:
        dir1: Saved trained scvi-tools model directory
        dir2: Saved trained scvi-tools model directory
        adata: AnnData used to train the models

    Returns:
        False if models are not equal, else True
    """
    model1 = CellCap.load(dir1, adata=adata, use_gpu=False)
    model2 = CellCap.load(dir2, adata=adata, use_gpu=False)

    if str(model1.module.state_dict()) != str(model2.module.state_dict()):
        return False

    for p1, p2 in zip(model1.module.parameters(), model2.module.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False

    return True
