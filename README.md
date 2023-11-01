# CellCap

CellCap is a variational autoencoder to model interpretable correspondence between cellular identity and perturbation response for single-cell data

## What

Attempt to extract interpretable biological insights from scRNA-seq perturbation 
experiments.

Ideally we will be able to learn something that a simple linear model using 
pseudobulk measurements could not.

## Why

We have many scRNA-seq experiments which apply chemical and genetic 
perturbations to cells. The idea is to learn from these perturbations.

Some in the field think that understanding perturbations could be a route to 
predicting drug efficacy (as in "find a drug perturbation that acts in the 
opposite way as a genetic perturbation or disease condition").

## How

We are attempting to build on the work of Lotfollahi 2022 (see below) to create 
a compositional perturbation variational autoencoder, with a few important 
differences.  The "linear" model for latent space perturbation arithmetic is 
limiting.  In the case of a linear decoder (which we want for interpretability), 
linear latent space arithmetic restricts us to completely linear models.  The 
real idea here is to allow complex arithmetic in the latent space, rather 
than additive models, in order to enable the use of an interpretable linear 
decoder.

Novelty:

- Interpretability via a linear decoder
- Complex interactions between cell state and perturbation via an attention 
mechanism

From an engineering standpoint, the code in this repository is built on top of 
[`scvi-tools`](https://scvi-tools.org).

## Related work

- [Lotfollahi ... Theis. "Learning interpretable cellular responses to 
complex perturbations in high-throughput screens." 
bioRxiv (2022)](https://www.biorxiv.org/content/10.1101/2021.04.14.439903v2)
- [Lotfollahi, Wolf, Theis. "scGen predicts single-cell perturbation 
responses." Nature Methods (2019)](https://www.nature.com/articles/s41592-019-0494-8)

## Who

Yang Xu

Stephen Fleming

Mehrtash Babadi

_Cellarium .. Methods Group .. Data Sciences Platform .. Broad Institute_
