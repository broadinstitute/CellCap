"""Utility functions"""

import numpy as np
import pandas as pd
from scipy import stats

import torch
import torch.nn.functional as F

import anndata
from scvi.data import synthetic_iid


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
    elif classname.find("Linear") != -1:
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


def downsample(
    adata,
    column="cell_type",
    random_state=None,
    min_cells=15,
    keep_small_categories=False,
):
    counts = adata.obs[column].value_counts(sort=False)
    min_size = min(counts[counts >= min_cells])
    sample_selection = None
    for sample, num_cells in counts.items():
        if num_cells <= min_cells:
            if keep_small_categories:
                sel = adata.obs.index.isin(adata.obs[adata.obs[column] == sample].index)
            else:
                continue
        else:
            sel = adata.obs.index.isin(
                adata.obs[adata.obs[column] == sample]
                .sample(min_size, random_state=random_state)
                .index
            )
        if sample_selection is None:
            sample_selection = sel
        else:
            sample_selection |= sel
    return adata[sample_selection].copy()


def cosine_distance(matrix, vector):
    dot_product = np.dot(matrix, vector)
    matrix_magnitude = np.linalg.norm(matrix, axis=1)
    vector_magnitude = np.linalg.norm(vector)
    cosine_distances = dot_product / (matrix_magnitude * vector_magnitude)

    return cosine_distances


def identify_top_perturbed_genes(pert_loading=None, prog_index=1):
    df = pert_loading.iloc[:, (prog_index - 1)].values
    zscore = stats.zscore(df)
    pval = stats.norm.sf(abs(zscore)) * 2
    ranked_df = pd.DataFrame.from_dict({"Zscore": zscore, "Pval": pval})
    ranked_df.index = pert_loading.index

    return ranked_df


def permute_dims(z):
    assert z.dim() == 2

    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)


def entropy(x, temp=1.0):
    p = F.softmax(x / temp, dim=1)  # + 1e-8
    logp = F.log_softmax(x / temp, dim=1)  # + 1e-8
    return -(p * logp).sum(dim=1)


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def cal_off_diagonal_corr(z):
    c = torch.matmul(z, z.T)
    off_diag = off_diagonal(c).pow_(2).sum()
    return off_diag


def _random_one_hot(n_classes: int, n_samples: int):
    # https://stackoverflow.com/questions/45093615/random-one-hot-matrix-in-numpy
    return np.eye(n_classes)[np.random.choice(n_classes, n_samples)]


def generate_simulated_dataset() -> anndata.AnnData:
    # random data via scvi-tools
    adata = synthetic_iid()
    n = adata.shape[0]
    del adata.obsm["protein_expression"]

    # add in necessary fields
    obsm_data = {
        "cond": _random_one_hot(2, n),
        "cont": _random_one_hot(2, n),
        "target": _random_one_hot(2, n),
        "donor": _random_one_hot(2, n),
    }
    for k, v in obsm_data.items():
        adata.obsm[k] = v
    adata.obs["pert"] = _random_one_hot(2, n)[:, 0]

    return adata
