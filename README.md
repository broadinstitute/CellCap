CellCap
==========

CellCap is a variational autoencoder for modeling correspondence between cellular identity and perturbation response
in single-cell data.

What is CellCap?
----------------

The primary goal of CellCap is
modeling the correspondence between cellular identity and perturbation response. To understand this correspondence,
CellCap was built in several explainable components:

![alt text](https://github.com/broadinstitute/CellCap/blob/main/docs/source/_static/design/Figure1.jpg?raw=false)

1. Basal state `z_basal`: basal state `z_basal` can be understood as cell state in latent space where only intrinsic
cellular identity is preserved.

2. Perturbation program `w`: each perturbation program has its latent representation `w` that has the same dimension as
basal state `z_basal` and explains the transcriptional activation or deactivation after perturbation.

3. Perturbation key `K`: matching a perturbation key `K` with a basal state `z_basal` through attention mechanism can
establish the correspondence between cellular identity and perturbation response. The output attention score `A`
further represents the relevance of each basal state `z_basal` to this perturbation program `w`.

4. Program relevance `H`: similar to attention score `A` telling the relevance of each basal state `z_basal` to a
perturbation program `w`, `H` presents the relevance of a perturbation condition to a perturbation program.

Is CellCap right for your data?
-------------------------------

Before you proceed to run CellCap, it's always important to ask if CellCap is right for your perturbation data. For
CellCap to accurately model the correspondence between cellular identity and perturbation response, there are a few
underlying assumptions to be considered:

1. The variation within unperturbed cells (control group) is primarily driven by cellular identity, and all possible
cell states should have their representatives in the control group.

2. After perturbation, cells still remain their cellular identities that are explained by cell states. Loss of
cellular identities due to perturbation would obscure the correspondence between cellular identity and perturbation
response.

3. Perturbation will induce transcriptional activation or deactivation, but this change will not establish new cell
states that are not present in control group.

4. If perturbation induces cells to establish new cell states, these perturbed cells should still preserve their old
cellular identities.

What biological investigations do CellCap aim to address?
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

Quick installation
------------------

CellCap can be installed via
    
    pip install cellcap
    

Citing CellCap
--------------

If you use CellCap in your research, please consider citing our paper:

Yang Xu, Stephen Fleming, Matthew Tegtmeyer, Steven A. McCarroll, and Mehrtash Babadi.
Modeling interpretable correspondence between cellular identity and perturbation response with CellCap.
*bioRxiv*, 2023.
