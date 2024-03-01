CellCap
==========

CellCap is a variational autoencoder for modeling correspondence between cell state and perturbation response in single-cell data.

Key concept of CellCap
-------------------------------
To understand such correspondence between cell state and perturbation response , CellCap was built with several explainable components:

![alt text](https://github.com/broadinstitute/CellCap/blob/main/docs/source/_static/design/Figure1.jpg?raw=false)

1. Basal state z<sub>nk</sub><sup>(basal)</sup>: basal state z<sub>nk</sub><sup>(basal)</sup> can be understood as cell state in latent space where only intrinsic
cellular identity is preserved.

2. Responsse program w<sub>qk</sub>: each response program p has its latent representation w<sub>k</sub> that has the same dimension as
basal state z<sup>(basal)</sup>, and it explains the transcriptional activation or deactivation after perturbation.

3. Perturbation key $\Kappa$<sub>pqk</sub>: matching a perturbation key `K` with a basal state z<sup>(basal)</sup> through attention mechanism can establish the correspondence between cell state and perturbation response. The output attention score $\beta$<sub>nq</sub> further represents the relevance of each basal state z<sup>(basal)</sup> to this perturbation program w<sub>k</sub>.

4. Program relevance H<sub>pq</sub>: similar to attention score `A` telling the relevance of each basal state z<sup>(basal)</sup> to a response program w<sub>k</sub>, H<sub>q</sub> presents the relevance of a perturbation condition p to a response program q.

Downstream analyses with CellCap
---------------------------------------------------------

Understanding the correspondence between cellular identity and perturbation response would facilitate multiple
biological investigations. We list a few suggested questions here, and users can use explainable components in CellCap
to address them.

![alt text](https://github.com/broadinstitute/CellCap/blob/main/docs/source/_static/design/Figure2.jpg?raw=false)

1. Do perturbations behave similarly or differently: Use of `H` alone could describe a general relationship between any
two perturbations. The same question can be asked again under a specific cellular context. Combining `A` and `H` can be
used to understand if different perturbations cause similar cellular responses in given a cell population.

2. What transcriptional responses can a perturbation induce: `H` also tells which perturbation programs `w` are relevant
to a perturbation. Each perturbation program `w` then aims to reveal a set of genes that are concurrently activated or
deactivated in perturbed cells.

3. Is a certain transcriptional response posed by a perturbation is cell-state specific: given perturbation program `w`
of interest, basal state `z_basal` can be sorted by matching perturbation key `K`. This is the key question CellCap
tries to address: the correspondence between cellular identity and perturbation response.

4. How large is this transcriptional response in this cell state: given a perturbed cell that was treated with a
perturbation, CellCap first infers its basal state `z_basal`. Then, multiplying `A` and `H` for this perturbed cell
can reveal how large perturbation program is in a cell state.

Navigating this Repository
------------------

The SynapseCLR repository is organized as follows:
```
<repo_root>/
├─ pytorch_synapse/       # SynapseCLR Python packages
├─ configs/               # Sample configuration files for pretraining SynapseCLR models
├─ scripts/               # Helper scripts
├─ notebooks/             # Notebooks for data preprocessing, interactive analysis, and reproducing paper figures
├─ data/                  # (not included in GitHub; see below) Raw and processed 3D EM image chunks
├─ ext/                   # (not included in GitHub; see below) External resources (e.g. other pretrained models)
├─ output/                # (not included in GitHub; see below) SynapseCLR outputs (pretrained models, extracted features, interactive analysis results)
└─ tables/                # (not included in GitHub; see below) Primary and derived resource tables
```

Preprint and Citation
--------------

The preprint will be shortly posted to *bioRxiv*, and this section will be updated imminently.
