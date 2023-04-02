"""Generate simulated data with known ground truth"""

import torch
import numpy as np
import scipy.sparse as sp
import pandas as pd
import scanpy as sc
import anndata
from typing import List, Tuple, Optional


def torch_normalize(x: torch.Tensor, n: float = 1e4) -> torch.Tensor:
    """Normalize counts per cell

    Args:
        x: count matrix of raw counts
        n: desired number of counts per cell

    Returns:
        normalized count matrix
    """
    return x / (x.sum(dim=-1, keepdim=True) + 1e-10) * n


def torch_scale(
    x: torch.Tensor,
    clip: float = 10.0,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """z-score count matrix per gene

    Args:
        x: count matrix, potentially normalized
        clip: clip z-scores at this value
        mean: if provided, this is used as the mean per gene
        std: if provided, this is used as the std per gene

    Returns:
        scaled count matrix
        mean per gene
        std per gene
    """
    if mean is None:
        mean = x.mean(dim=0, keepdim=True)
    x = x - mean.to(x.device)
    if std is None:
        std = x.std(dim=0, keepdim=True)
        std = torch.where(std == 0, torch.ones_like(std), std)
    x = x / std.to(x.device)
    return torch.clamp(x, min=-np.abs(clip), max=np.abs(clip)), mean, std


def torch_pca(x: torch.Tensor, n_pcs: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform PCA using pytorch and return top n_pcs

    Args:
        x: matrix where rows are observations and columns are features
        n_pcs: number of PCs to keep

    Returns:
        transformed_data: matrix where rows are observations and columns are top n_pcs
        pca_weight_matrix: matrix that gives the transformation to PC space
    """

    U, S, V = torch.pca_lowrank(x, q=n_pcs)
    pca_weight_matrix_gm = V[:, :n_pcs]
    transformed_data = torch.matmul(x, pca_weight_matrix_gm)
    return transformed_data, pca_weight_matrix_gm


def get_posterior_z_and_trained_decoder(
    adata: anndata.AnnData,
    layer: Optional[str] = None,
    n_genes: int = 1000,
    n_latent: int = 10,
) -> Tuple[torch.Tensor, torch.nn.Module, pd.DataFrame]:
    """Train a model and obtain the decoder and posterior latent z

    Since we are interested here exclusively in linear decoders, one way to get
    this kind of posterior and decoder is just to do PCA!

    Args:
        adata: real single cell data
        layer: count data should be in this layer or None to use adata.X
        n_genes: number of genes (highly variable) to use
        n_latent: size of the latent dimension

    Returns:
        z_basal
        decoder
    """

    adata = adata.copy()
    sc.pp.highly_variable_genes(
        adata, layer=layer, flavor="seurat_v3", n_top_genes=n_genes
    )
    adata = adata[:, adata.var["highly_variable"]]  # subset this thing

    # get count matrix as dense tensor
    if layer is None:
        x = adata.X
    else:
        x = adata.layers[layer]
    if sp.issparse(x):
        x = x.todense()
    x_ng = torch.from_numpy(x)

    # normalize
    x_norm_ng = torch_normalize(x_ng)

    # scale
    x_norm_ng, _, _ = torch_scale(x_norm_ng)

    # do PCA
    z_nm, _ = torch_pca(x_norm_ng, n_pcs=n_latent)

    # scale the latent space to approx unit norm
    norm = torch.square(z_nm).sum(dim=-1).sqrt().mean()
    z_nm = z_nm / norm

    # get the transformation as a torch module
    z_mm = z_nm[:n_latent, :]
    x_mg = x_ng[:n_latent, :]
    x_mean_mg = x_mg.mean(dim=0)
    x_centered_mg = x_mg - x_mean_mg
    decoder = torch.nn.Linear(in_features=n_latent, out_features=n_genes, bias=True)
    decoder.weight = torch.nn.Parameter(torch.linalg.solve(z_mm, x_centered_mg).t())
    decoder.bias = torch.nn.Parameter(x_mean_mg)

    return z_nm, decoder, adata.var


def simulate_data_from_real_data(
    adata: anndata.AnnData,
    layer: Optional[str] = None,
    n_cells_per_perturbation: List[int] = [500, 500, 500],
    n_cells_control: int = 1000,
    n_latent: int = 10,
    n_genes: int = 1000,
    n_donors: int = 2,
    donor_scale: float = 0.1,
    donor_mixing_in_basal_space: float = 0.9,
    donor_representation_concentration: float = 10.0,
    response_program_matrix: torch.Tensor = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
        ]
    ),
    perturbation_response_matrix: torch.Tensor = torch.tensor(
        [[0, 0, 0, 0.5, 0.1], [0.8, 0.1, 0, 0, 0], [0.5, 0, 0.3, 0.7, 0]]
    ),
    maturity_gradient_scale: float = 1.0,
    include_interactions: bool = True,
    donor_interaction_scale: bool = 0.2,
    basal_interaction_scale: bool = 1.5,
    random_seed: int = 0,
) -> anndata.AnnData:
    """Generate a simulated dataset, where z_basal is based on real data

    Args:
        adata: AnnData object containing real data. Makes most sense to have all
            cells come from the control condition. Can be several cell types.
            Probably just use one donor or one batch.
        layer: Count data should be in this layer of AnnData.
            Use None for adata.X
        n_cells_per_perturbation: Number of cells to simulate for each
            perturbed condition
        n_cells_control: Number of control cells
        n_latent: Dimension of latent space
        n_genes: Number of genes simulated
        n_donors: Number of donors
        donor_scale: Scale factor for donor effect (compared to 1)
        donor_mixing_in_basal_space: In (0, 1]. 1 for well-mixed, less than
            1 for donor differences in the basal space
        donor_representation_concentration: Dirichlet concentration parameter
            used to draw the numbers of cells for each donor. Small values < 1
            will give uneven draws, while large numbers > 1 will be very even.
        response_program_matrix: [response programs, latent dimensions]
        perturbation_response_matrix: [perturbations, response programs]
        maturity_gradient_scale: Values > 0 will model a gradient of cell
            maturity. 1 is a large gradient. This gradient controls interaction
            effects.
        include_interactions: True to model basal-state dependence of the donor
            effect and of the perturbation effect
        donor_interaction_scale: Scale of the donor-dependent modulation of the
            effects of perturbations. In [0, 1].
        basal_interaction_scale: Scale of the basal-state-dependent modulation
            of the effects of perturbations, >= 0, where interaction is with the
            maturity gradient in the basal space
        random_seed: Seed for random number generator

    Returns:
        Simulated AnnData object that contains truth data
    """

    n_perturbations = len(n_cells_per_perturbation)

    assert (donor_mixing_in_basal_space <= 1.0) and (
        donor_mixing_in_basal_space > 0
    ), "donor_mixing_in_basal_space must be <= 1 and > 0"

    assert (donor_interaction_scale <= 1.0) and (
        donor_interaction_scale >= 0
    ), "donor_interaction_scale must be <= 1 and >= 0"

    assert basal_interaction_scale >= 0, "basal_interaction_scale must be >= 0"

    if (basal_interaction_scale > 0) and (maturity_gradient_scale == 0):
        print(
            "Warning: need to set maturity_gradient_scale > 0 if you want "
            "to model basal-state dependent perturbation response strengths"
        )

    assert donor_scale >= 0.0, "donor_scale must be >= 0"

    assert (maturity_gradient_scale <= 1.0) and (
        maturity_gradient_scale >= 0
    ), "maturity_gradient_scale must be >= 0 and <= 1"

    assert (
        response_program_matrix.shape[1] == n_latent
    ), "response_program_matrix.shape[1] must be equal to n_latent"

    assert (
        perturbation_response_matrix.shape[0] == n_perturbations
    ), "length of n_cells_per_perturbation must match perturbation_response_matrix.shape[0]"

    assert (
        response_program_matrix.shape[0] == perturbation_response_matrix.shape[1]
    ), "response_program_matrix.shape[0] must match perturbation_response_matrix.shape[1]"

    torch.random.manual_seed(seed=random_seed)

    n_cells = np.sum(n_cells_per_perturbation) + n_cells_control

    # normalize the response programs
    response_program_norms = (
        response_program_matrix.pow(2).sum(dim=-1, keepdim=True).sqrt()
    )
    response_program_matrix = response_program_matrix / response_program_norms

    # determine cell conditions: donor and perturbation
    n_cells_per_donor = (
        torch.distributions.Dirichlet(
            concentration=torch.tensor(
                [donor_representation_concentration] * n_donors
            ).float(),
        ).sample()
        * n_cells
    )
    n_cells_per_donor = n_cells_per_donor.round()
    n_cells_per_donor[-1] = n_cells - n_cells_per_donor[:-1].sum()
    n_cells_per_donor = n_cells_per_donor.int().tolist()
    donors_n = torch.tensor(
        sum(
            [[i] * n for i, n in enumerate(n_cells_per_donor)],
            [],  # https://stackoverflow.com/a/716489/19753230
        )
    ).long()
    donors_nd = torch.nn.functional.one_hot(donors_n).float()
    perturbation_n = torch.tensor(
        sum(
            [
                [i] * n
                for i, n in enumerate([n_cells_control] + n_cells_per_perturbation)
            ],
            [],  # https://stackoverflow.com/a/716489/19753230
        )
    ).long()
    perturbation_np = torch.nn.functional.one_hot(perturbation_n).float()

    # get z_basal and the decoder from a real dataset
    z_basal__m, decoder, var = get_posterior_z_and_trained_decoder(
        adata=adata,
        layer=layer,
        n_genes=n_genes,
        n_latent=n_latent,
    )
    jitter_scale = 0.1
    z_basal_nm = z_basal__m + torch.randn(z_basal__m.size()) * jitter_scale
    while z_basal_nm.shape[0] < n_cells:
        z_basal_nm = torch.cat(
            [z_basal_nm, z_basal__m + torch.randn(z_basal__m.size()) * jitter_scale]
        )
    z_basal_nm = z_basal_nm[:n_cells]

    # blur z_basal a bit
    z_basal_nm = z_basal_nm + torch.randn([n_cells, n_latent]) * 0.2

    # optionally assign a cell differentiation maturity gradient to z_basal
    reference_vector_m = torch.zeros([n_latent])
    reference_vector_m[0] = -1.0
    reference_vector_m[1] = 1.0
    reference_vector_m[3] = -1.0
    dist_n = -1 * (reference_vector_m.unsqueeze(0) - z_basal_nm).pow(2).sum(dim=-1)
    dist_n = dist_n - np.percentile(
        dist_n.numpy(), q=max(5.0, 100.0 - maturity_gradient_scale * 100)
    )
    dist_n = torch.clamp(dist_n, min=0)
    delta_maturity_scale_n = dist_n / dist_n.max() * maturity_gradient_scale

    # optionally make z_basal a bit different for different donors
    maturity_vector_m = (
        z_basal_nm[delta_maturity_scale_n.argmax()]
        - z_basal_nm[delta_maturity_scale_n.argmin()]
    )
    maturity_direction_m = maturity_vector_m / maturity_vector_m.sum()
    for d, m in zip(np.unique(donors_n), np.linspace(0, 1, n_donors)):
        # draw a mixing fraction for each donor
        logic = donors_n == d
        a = 1.0 - m * (1.0 - donor_mixing_in_basal_space)
        z_basal_nm[logic, :] = a * z_basal_nm[logic, :] + (
            1.0 - a
        ) * maturity_direction_m.unsqueeze(0)

    # simulate the additive contribution for z_donor
    delta_z_donor_dm = torch.randn([n_donors, n_latent])
    delta_z_donor_dm = (
        delta_z_donor_dm / delta_z_donor_dm.sum(dim=-1, keepdim=True) * donor_scale
    )
    delta_z_donor_nm = torch.matmul(donors_nd, delta_z_donor_dm)

    # simulate the additive perturbation effect
    delta_z_perturbation_pm = torch.cat(
        [
            torch.zeros([1, n_latent]),
            torch.matmul(
                perturbation_response_matrix.float(),
                response_program_matrix.float(),
            ),
        ]
    )
    delta_z_perturbation_nm = torch.matmul(perturbation_np, delta_z_perturbation_pm)

    # interaction terms which make attention necessary
    if include_interactions:
        # simulate the donor-dependent modulation of the perturbation effect
        donor_effect_d = 1.0 - torch.rand([n_donors]) * donor_interaction_scale
        donor_effect_n1 = torch.matmul(donors_nd, donor_effect_d.unsqueeze(-1))

        # simulate the basal-state-dependent modulation of the perturbation effect
        # maturity_z_score_n = delta_maturity_scale_n
        # maturity_z_score_n = maturity_z_score_n - maturity_z_score_n.mean()
        # maturity_z_score_n = maturity_z_score_n / maturity_z_score_n.std()
        # basal_effect_n1 = (maturity_z_score_n * basal_interaction_scale + 1.).unsqueeze(-1)
        norm_maturity_n = delta_maturity_scale_n / np.percentile(
            delta_maturity_scale_n.numpy(), q=95
        )
        basal_effect_n = torch.clamp(
            norm_maturity_n + (1.0 - basal_interaction_scale), min=0, max=1
        )
        basal_effect_n = basal_effect_n / basal_effect_n.max()
        basal_effect_n1 = basal_effect_n.unsqueeze(-1)

    else:
        donor_effect_n1 = 1.0
        basal_effect_n1 = 1.0

    # final z
    z_nm = (
        z_basal_nm
        # + delta_z_donor_nm
        + (delta_z_perturbation_nm * donor_effect_n1 * basal_effect_n1)
    )

    # gene expression
    x_ng = decoder(z_nm).detach()

    # put everything in adata
    adata_sim = anndata.AnnData(
        X=x_ng.numpy(),
        obs=None,
        var=var,
    )
    adata_sim.obs["donor"] = donors_n.numpy()
    adata_sim.obs["donor"] = adata_sim.obs["donor"].astype("category")
    adata_sim.obs["perturbation"] = perturbation_n.numpy()
    adata_sim.obs["perturbation"] = adata_sim.obs["perturbation"].astype(str)
    adata_sim.obs["perturbation"] = adata_sim.obs["perturbation"].apply(
        lambda s: s.replace("0", "control")
    )
    adata_sim.obs["perturbation"] = adata_sim.obs["perturbation"].astype("category")
    adata_sim.obs["maturity_norm"] = delta_maturity_scale_n.numpy()
    adata_sim.obs["basal_interaction"] = basal_effect_n1.squeeze().numpy()
    adata_sim.obs["donor_interaction"] = donor_effect_n1.squeeze().numpy()
    adata_sim.uns["response_program_matrix"] = response_program_matrix.numpy()
    adata_sim.uns["perturbation_response_matrix"] = perturbation_response_matrix.numpy()
    adata_sim.uns["decoder_weight"] = decoder.weight.detach().numpy()
    adata_sim.uns["decoder_bias"] = decoder.bias.detach().numpy()
    adata_sim.obsm["delta_z_donor"] = delta_z_donor_nm.numpy()
    adata_sim.obsm["delta_z_perturbation"] = delta_z_perturbation_nm.numpy()
    adata_sim.obsm["z_basal"] = z_basal_nm.numpy()
    adata_sim.obsm["z"] = z_nm.numpy()

    return adata_sim
