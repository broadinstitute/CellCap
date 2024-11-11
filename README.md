CellCap
=======

CellCap is a generative model of scRNA-seq perturbation data which emphasizes interpretability. CellCap explicitly models the correspondence between each cell's basal state and the measured perturbation response, and learns how to explain cellular response in terms of weighted sums of a succinct set of response programs. A detailed <a href="https://cellcap.readthedocs.io/en/latest/">documentation</a> is also available.

Key concepts of CellCap
-----------------------
To understand the correspondence between basal cell state and perturbation response, the CellCap model was built with several interpretable components:

![Fig.1a-b](https://github.com/broadinstitute/CellCap/blob/main/docs/source/_static/design/Figure1.jpg?raw=false)

1. Basal state $z_{nk}^\text{(basal)}$: this basal state of each cell $n$ can be understood as the pre-perturbation cell state in latent space $k$, where only intrinsic cellular variations are preserved. This basal state does not contain information about the applied perturbation or other covariate information (e.g. batch, donor, ...).

2. Response programs $w_{qk}$: each response program $q$ has a latent representation $\vec{w}_k$ that explains the transcriptional activation or deactivation of sets of genes in response to a perturbation.

3. Attention mechanism: shown in the blue rectangle in panel (a), a multi-head, scaled dot-product attention mechanism is used to model the correspondence between a basal cell state $z_{nk}^\text{(basal)}$ and perturbation responses $w_{qk}$. CellCap learns a set of key vectors $\kappa_{pqk}$ to capture this correspondence. The attention weights $\beta_{nq}$ represent the relevance of each response program $q$ to each basal cell state $z_{nk}^\text{(basal)}$.

4. Program relevance $H_{pq}$: a weight matrix that captures the relevance of response program $q$ to perturbation condition $p$. Bayesian automatic relevance determination is used by the model to keep this matrix sparse and interpretable.

Downstream analyses with CellCap
--------------------------------

Understanding the correspondence between cellular identity and perturbation response can help answer many biological questions. We list a few suggested questions here which can be answered using the key concepts of CellCap above.

![Fig.1c-f](https://github.com/broadinstitute/CellCap/blob/main/docs/source/_static/design/Figure2.jpg?raw=false)

1. How do perturbations relate to each other?

    General relationships are captured by the learned program relevance matrix $H_{pq}$. Additionally, the per-cell response matrix $h_{nq}$ can be used to understand whether different perturbations cause similar responses in given a cell population.

2. What are the transcriptional response programs that we see in a dataset?

    Each response program $q$ in the learned response program matrix $w_{qk}$ aims to reveal a set of genes that are coherently activated or deactivated in perturbed cells. Translating $w_{qk}$ from latent space to gene expression space is achieved by sending $w_{qk}$ through CellCap's linear decoder.

3. For a given perturbation, which cell states respond strongly via each response program?

    For one perturbation $p$, given a response program $q$ of interest, the basal state $z_{nk}^\text{(basal)}$ can be queried by the learned perturbation key $\kappa_{pqk}$. This is a key question that CellCap addresses: the correspondence between basal cell state and perturbation response.

4. What is the response amplitude of each response program in each basal cell state?

    CellCap infers response amplitude "attention weights" $\beta_{nq}$ for each response program $q$ in each cell $n$ given its learned basal state $z_{nk}^\text{(basal)}$.

Navigating this Repository
--------------------------

The CellCap repository is organized as follows:
```
<repo_root>/
├─ cellcap/               # CellCap python package
└─ docs/                  # Package documentation
    └─ source/
        └─ notebooks/     # Example jupyter notebooks
```

Installation
------------
We suggest creating a new conda environment to run CellCap

```
conda create -n cellcap python=3.10
conda activate cellcap

git clone https://github.com/broadinstitute/CellCap.git
cd CellCap

pip install .
```

Preprint and Citation
---------------------

If you use CellCap in your research, please consider citing our paper:

Yang Xu, Stephen Fleming, Matthew Tegtmeyer, Steven A. McCarroll, Mehrtash Babadi. Modeling interpretable correspondence between cell state and perturbation response with CellCap. [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.03.14.585078v1), 2024.
