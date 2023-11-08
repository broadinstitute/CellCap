.. _introduction:

What is CellCap?
===================

CellCap is a variational autoencoder for analyzing single-cell perturbation data. The primary goal of CellCap is
modeling the correspondence between cellular identity and perturbation response. CellCap also has a few explainable
components to facilitate understanding this correspondence:

1. Basal state `z_basal`: we can understand basal state `z_basal` as cell state in latent space that only intrinsic
cellular identity is preserved.

2. Perturbation program `w`: each perturbation program has its latent representation `w` that explains the
transcriptional change after perturbation.

3. Perturbation key `K`: matching perturbation key `K` with basal state `z_basal` through attention mechanism can
establish a correspondence between cellular identity and perturbation response. The output attention score `A`
further represents the relevance of each perturbation program `w` to this basal state `z_basal`.

4. Program relevance `H`: similar to attention score `A` telling the relevance of a perturbation program `w` to a basal
state `z_basal`, `H` presents the relevance of a perturbation program to a perturbation condition.

Is CellCap right for your data?
-----------------

Before you proceed to run CellCap, it's always important to ask if CellCap is right for your perturbation data. For
CellCap to accurately model the correspondence between cellular identity and perturbation response, there are a few
underlying assumptions:

1. The variation within unperturbed cells (control group) is primarily driven by cellular identity, and all possible
cell states have their representatives in the control group.

2. After perturbation, cells still preserve their cellular identities that are explained by cell states. Loss of
cellular identities due to perturbation would obscure the correspondence between cellular identity and perturbation
response.

3. Perturbation will induce transcriptional activation or deactivation, but this change will not establish new cell
states that are not present in control group.

4. If perturbation induces cells to establish new cell states, these perturbed cells should still preserve their old
cellular identities.

Use CellCap to probe biological investigations
-------

Understanding the correspondence between cellular identity and perturbation response would facilitate multiple
biological investigations. We list a few suggested questions here, and users can use explainable components in CellCap
to address them.

1. Do perturbations behave similarly or differently: Use of `H` alone could describe a general relation between any two
perturbations. The same question can be asked under a specific cellular context. Combining `A` and `H` can be used
to understand if different perturbations cause similar cellular responses in given a cell population.

2. What transcriptional responses can a perturbation induce:

3. Is a certain transcriptional response posed by a perturbation is cell-state specific:

4. How large is this transcriptional response in this cell state:

More information
-------

For detailed scope and discussion about CellCap, please see our manuscript.
