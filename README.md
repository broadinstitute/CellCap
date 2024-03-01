CellCap
=======

CellCap is a variational autoencoder for modeling interpretable correspondence between cell state and perturbation response in single-cell data.

Key concept of CellCap
----------------------
To understand such correspondence between cell state and perturbation response , CellCap was built with several explainable components:

![alt text](https://github.com/broadinstitute/CellCap/blob/main/docs/source/_static/design/Figure1.jpg?raw=false)

1. Basal state z<sub>nk</sub><sup>(basal)</sup>: basal state z<sub>nk</sub><sup>(basal)</sup> can be understood as cell state in latent space where only intrinsic
cellular identity is preserved.

2. Responsse program w<sub>qk</sub>: each response program p has its latent representation w<sub>k</sub> that has the same dimension as
basal state z<sup>(basal)</sup>, and it explains the transcriptional activation or deactivation after perturbation.

3. Perturbation key $\kappa$<sub>pqk</sub>: matching a key $\kappa$<sub>qk</sub> of perturbation p with a basal state z<sup>(basal)</sup> through attention mechanism can establish the correspondence between cell state and perturbation response. The output attention score $\beta$<sub>nq</sub> further represents the relevance of each basal state z<sup>(basal)</sup> to a response program q.

4. Program relevance H<sub>pq</sub>: similar to attention score $\beta$<sub>nq</sub> telling the relevance of each basal state z<sup>(basal)</sup> to a response program q, H<sub>q</sub> presents the relevance of a perturbation condition p to a response program q.

Downstream analyses with CellCap
--------------------------------

Understanding the correspondence between cellular identity and perturbation response would facilitate multiple
biological investigations. We list a few suggested questions here, and users can use explainable components in CellCap
to address them.

![alt text](https://github.com/broadinstitute/CellCap/blob/main/docs/source/_static/design/Figure2.jpg?raw=false)

1. Do perturbations behave similarly or differently: Use of H<sub>pq</sub> alone could describe a general relationship between any two perturbations. The same question can be asked again under a specific cellular context. Combining $\beta$<sub>nq</sub> and H<sub>pq</sub> can be used to understand if different perturbations cause similar cellular responses in given a cell population.

2. What transcriptional responses can a perturbation induce: Each perturbation program q aims to reveal a set of genes that are concurrently activated or deactivated in perturbed cells. This can be done by forwarding w<sub>qk</sub> through the linear decoder.

3. Is a certain transcriptional response posed by a perturbation is cell-state specific: given a response program q
of interest, basal state z<sub>nk</sub><sup>(basal)</sup> can be sorted by matching perturbation key $\kappa$<sub>pqk</sub>. This is the key question CellCap
tries to address: the correspondence between cell state and perturbation response.

4. How large is this transcriptional response in this cell state: given a perturbed cell that was treated with a
perturbation p, CellCap first infers its basal state z<sup>(basal)</sup>. Then, CellCap infers response amplitude $\beta$<sub>q</sub> for this perturbed cell
given its basal stata z<sup>(basal)</sup>.

Navigating this Repository
--------------------------

The CellCap repository is organized as follows:
```
<repo_root>/
├─ cellcap/               # CellCap python package
├─ doc/                   # Package documentation, including notebooks for analyzing single-cell perturbation data with CellCap
```

Preprint and Citation
--------------

The preprint will be shortly posted to *bioRxiv*, and this section will be updated imminently.
