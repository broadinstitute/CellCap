"""Validation plotting functions"""

import anndata
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .stats import compute_basal_state_classifier_stats
from ..scvi_module import CellCap


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
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
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


def plot_ard_parameters(
    cellcap: CellCap,
    adata: anndata.AnnData,
    perturbation_key: str = "perturbation",
) -> pd.DataFrame:
    """Plot a matrixplot showing the ARD parameter for each (response program, perturbation)

    Args:
        cellcap: The trained model
        adata: AnnData object
        perturbation_key: Key of adata.obs that contains the perturbation information

    Returns:
        Summary dataframe with the numerical values of the response program usage

    """

    ard = cellcap.get_ard()

    im = plt.imshow(ard, cmap="Oranges", vmin=0, vmax=1)
    plt.grid(False)
    plt.xticks(ticks=range(ard.shape[1]))
    plt.xlabel("Response program")

    # TODO: once the input format is updated to ObsField instead of ObsmField,
    # TODO: then we can pull this label information more directly from the model
    plt.yticks(
        ticks=range(ard.shape[0]),
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
