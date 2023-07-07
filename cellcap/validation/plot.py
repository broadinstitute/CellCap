"""Validation plotting functions"""

import anndata
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .stats import (
    compute_basal_state_classifier_stats,
    compute_basal_state_regression_stats,
)
from ..scvi_module import CellCap

from typing import Dict


def plot_adversarial_classifier_roc(
    adata: anndata.AnnData,
    perturbation_key: str = "perturbation",
    basal_key: str = "X_basal",
    verbose: bool = True,
):
    """Plot a ROC curve for a (newly trained) classifier which attempts to
    predict the perturbation from the basal state.

    Args:
        adata: AnnData
        perturbation_key: Key of adata.obs that contains perturbation information
        basal_key: Key of adata.obsm that contains the learned basal state
        verbose: True to print the AUC information for the classifier

    """

    stats = compute_basal_state_classifier_stats(
        adata=adata,
        perturbation_key=perturbation_key,
        basal_key=basal_key,
    )

    unique_perturbation_conditions = stats["unique_perturbation_conditions"]
    fpr = stats["fpr"]
    tpr = stats["tpr"]
    roc_auc = stats["auc"]

    colors = list(sns.color_palette("Paired")) + list(sns.color_palette("hls", 8))

    plt.figure(figsize=(5, 5))
    for i, color in zip(range(adata.obs[perturbation_key].nunique()), colors):
        plt.plot(
            fpr[unique_perturbation_conditions[i]],
            tpr[unique_perturbation_conditions[i]],
            color=color,
            lw=1,
            label="{0}".format(unique_perturbation_conditions[i]),
        )
    plt.grid(False)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("")
    legend = plt.legend(loc="lower right", prop={"size": 15})
    legend.get_frame().set_facecolor("none")

    if verbose:
        for i, color in zip(range(len(unique_perturbation_conditions)), colors):
            print(
                "{0} (AUC = {1:0.4f})".format(
                    unique_perturbation_conditions[i],
                    roc_auc[unique_perturbation_conditions[i]],
                )
            )


def plot_program_usage(
    cellcap: CellCap,
    adata: anndata.AnnData,
    perturbation_key: str = "perturbation",
) -> pd.DataFrame:
    """Plot a matrixplot showing the usage of each response program by each perturbation

    Args:
        cellcap: The trained model
        adata: AnnData object
        perturbation_key: Key of adata.obs that contains the perturbation information

    Returns:
        Summary dataframe with the numerical values of the response program usage

    """

    h_attn = cellcap.get_h_attn()
    df = pd.DataFrame(h_attn, index=adata.obs_names)
    df["condition"] = adata.obs[perturbation_key].copy()
    h_attn_per_perturbation = df.groupby("condition").mean()

    im = plt.imshow(h_attn_per_perturbation.to_numpy(), cmap="Oranges", vmin=0)
    plt.grid(False)
    plt.xticks(ticks=range(h_attn.shape[1]))
    plt.xlabel("Response program")
    plt.yticks(
        ticks=range(len(h_attn_per_perturbation)),
        labels=h_attn_per_perturbation.index,
    )
    plt.ylabel("Perturbation")
    plt.title("Program usage")
    plt.colorbar(im, fraction=0.025, pad=0.04)

    return h_attn_per_perturbation


def plot_program_usage_ignoring_attention(
    cellcap: CellCap,
    adata: anndata.AnnData,
    perturbation_key: str,
) -> pd.DataFrame:
    """Plot a matrixplot showing the usage of each response program by each
    perturbation, if attention were not taken into account.

    Args:
        cellcap: The trained model
        adata: AnnData object
        perturbation_key: Key of adata.obs that contains the perturbation information

    Returns:
        Summary dataframe with the numerical values of h_pq

    """

    h_pq = cellcap.get_h()

    im = plt.imshow(h_pq, cmap="Oranges", vmin=0)
    plt.grid(False)
    plt.xticks(ticks=range(h_pq.shape[1]))
    plt.xlabel("Response program")

    # TODO: once the input format is updated to ObsField instead of ObsmField,
    # TODO: then we can pull this label information more directly from the model
    plt.yticks(
        ticks=range(h_pq.shape[0]),
        labels=pd.get_dummies(
            (
                adata.obs[perturbation_key]
                .str.lower()
                .replace(to_replace="control", value=np.nan)
            )
        ).columns,
    )

    plt.ylabel("Perturbation")
    plt.title("Program usage\nwithout attention")
    plt.colorbar(im, fraction=0.025, pad=0.04)

    return h_pq


def plot_ard_parameters(
    cellcap: CellCap,
    adata: anndata.AnnData,
    perturbation_key: str = "perturbation",
) -> Dict[str, np.ndarray]:
    """Plot a matrixplot showing the ARD parameter for each (response program, perturbation)

    Args:
        cellcap: The trained model
        adata: AnnData object
        perturbation_key: Key of adata.obs that contains the perturbation information

    Returns:
        Summary dict

    """

    ard = cellcap.get_ard()

    im = plt.imshow(ard["local"], cmap="Oranges", vmin=0, vmax=1)
    plt.grid(False)
    plt.xticks(ticks=range(ard["local"].shape[1]))
    plt.xlabel("Response program")

    # TODO: once the input format is updated to ObsField instead of ObsmField,
    # TODO: then we can pull this label information more directly from the model
    plt.yticks(
        ticks=range(ard["local"].shape[0]),
        labels=pd.get_dummies(
            (
                adata.obs[perturbation_key]
                .str.lower()
                .replace(to_replace="control", value=np.nan)
            )
        ).columns,
    )

    plt.ylabel("Perturbation")
    plt.title("Program ARD relevance")
    plt.colorbar(im, fraction=0.025, pad=0.04)

    return ard


def plot_basal_correlation_with_attention(
    adata: anndata.AnnData,
    attention_key: str,
    basal_key: str = "X_basal",
) -> Dict[str, np.ndarray]:
    """Compute a linear regression between z_basal and the attention values
    in adata.obs[attention_key]

    Args:
        adata: AnnData object
        attention_key: Key of adata.obs that contains the perturbation information
        basal_key: Key of adata.obsm that contains the learned basal state

    Returns:
        Summary dict
    """

    stats = compute_basal_state_regression_stats(
        adata=adata,
        basal_key=basal_key,
        attention_key=attention_key,
    )

    x = np.array(stats["train_true"].tolist() + stats["test_true"].tolist())
    y = np.array(stats["train_predicted"].tolist() + stats["test_predicted"].tolist())
    c = np.array(
        len(stats["train_true"]) * ["train"] + len(stats["test_true"]) * ["test"]
    )
    order = np.argsort(x)

    plt.plot(
        x[order][c[order] == "train"],
        y[order][c[order] == "train"],
        "k.",
        label="train",
    )
    plt.plot(
        x[order][c[order] == "test"],
        y[order][c[order] == "test"],
        "r.",
        label="test",
    )
    plt.plot([0, 1], [0, 1], "-", color="lightgray", label="truth")
    plt.legend()
    plt.xlabel(f'True adata.obs["{attention_key}"] value')
    plt.ylabel("Value predicted from z_basal")
    plt.title(
        f'Variance explained on test data:\n{stats["test_variance_explained"]:.2f}'
    )

    return stats


def plot_learned_program_relationships(
        cellcap: CellCap,
        ard_threshold: float = 0.5,
):
    """Plot relationships (via scatterplots) between learned response programs,
    both in terms of usage of latent space and in terms of effect on gene expression.

    Args:
        cellcap: The trained model
        ard_threshold: Threshold to determine which response programs are significant,
            and only significant ones will be plotted. Value 0 will plot all.
    """

    ard = cellcap.get_ard()

    significant_program_inds = np.where(ard["global"] > ard_threshold)[0]

    for i in significant_program_inds:
        for j in significant_program_inds:
            if i == j:
                continue
            if j < i:
                continue

            plt.figure(figsize=(8, 4))

            plt.subplot(1, 2, 1)

            plt.plot(cellcap.module.w_qk[i, :].detach().cpu(), cellcap.module.w_qk[j, :].detach().cpu(), '.')
            plt.plot([-3, 3], [-3, 3], '-', color='lightgray')
            plt.xlabel(f'Program {i}')
            plt.ylabel(f'Program {j}')
            plt.title('Latent space weights in w_qk')
            plt.grid(False)

            plt.subplot(1, 2, 2)

            library = torch.tensor([7.]).to(device=cellcap.module.w_qk.device)

            exp_i = cellcap.module.decoder(
                dispersion=cellcap.module.dispersion,
                z=cellcap.module.w_qk[i, :].unsqueeze(0),
                library=library,
            )[-1].squeeze().detach().cpu()
            exp_j = cellcap.module.decoder(
                dispersion=cellcap.module.dispersion,
                z=cellcap.module.w_qk[j, :].unsqueeze(0),
                library=library,
            )[-1].squeeze().detach().cpu()

            plt.loglog(exp_i / exp_i.sum(), exp_j / exp_j.sum(), '.')
            plt.plot([0, 0.1], [0, 0.1], '-', color='lightgray')
            plt.xlabel(f'Program {i}')
            plt.ylabel(f'Program {j}')
            plt.title('Gene expression in program')
            plt.grid(False)

            plt.tight_layout()
