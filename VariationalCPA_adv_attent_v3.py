#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 08:26:14 2022

@author: yxu
"""

import logging
import numpy as np
import pandas as pd
from anndata import AnnData

import torch
from torch import Tensor
from torch import logsumexp
import torch.nn.functional as F
from torch.distributions import Normal, Poisson
from torch.distributions import kl_divergence as kl

from scvi import REGISTRY_KEYS
from scvi._compat import Literal
from scvi.train import TrainRunner
from scvi.train import TrainingPlan
from scvi.dataloaders import DataSplitter
from scvi.train._callbacks import SaveBestState
from scvi.nn import Encoder, LinearDecoderSCVI, one_hot
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.distributions import ZeroInflatedNegativeBinomial, NegativeBinomial
from scvi.model.base import UnsupervisedTrainingMixin, BaseModelClass, RNASeqMixin, VAEMixin

from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
    ObsmField,
)
from scvi.data import AnnDataManager
from scvi.utils import setup_anndata_dsp

logger = logging.getLogger(__name__)

from easydl import aToBSheduler
from typing import Callable, Iterable, Optional, List, Union, Tuple

torch.backends.cudnn.benchmark = True

def entropy(x,temp=1.0):
    p = F.softmax(x/temp, dim=1)# + 1e-8
    logp = F.log_softmax(x/temp,dim=1)# + 1e-8
    return -(p*logp).sum(dim=1)#.mean()

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)

#Gradiant reverse
class GradientReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coeff, input):
        ctx.coeff = coeff
        return input

    @staticmethod
    def backward(ctx, grad_outputs):
        coeff = ctx.coeff
        return None, -coeff * grad_outputs

class GradientReverseModule(torch.nn.Module):
    def __init__(self, scheduler):
        super(GradientReverseModule, self).__init__()
        self.scheduler = scheduler
        self.global_step = 0.0
        self.coeff = 0.0
        self.grl = GradientReverseLayer.apply
    def forward(self, x):
        self.coeff = self.scheduler(self.global_step)
        self.global_step += 1.0
        return self.grl(self.coeff, x)

class AdvNet(torch.nn.Module):
    def __init__(self, in_feature=20, hidden_size=20,out_dim=2):
        super(AdvNet, self).__init__()
        self.ad_layer1 = torch.nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = torch.nn.Linear(hidden_size, out_dim)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.norm1 = torch.nn.BatchNorm1d(hidden_size)
        self.norm2 = torch.nn.BatchNorm1d(hidden_size)
        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.25)
        self.sigmoid = torch.nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, 
                                                                   gamma=10, 
                                                                   max_iter=self.max_iter))

    def forward(self, x, reverse = True):
        if reverse:
            x = self.grl(x)
        x = self.ad_layer1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1

class DotProductAttention(torch.nn.Module):
    """
    Compute the dot products of the query with all values and apply a 
    softmax function to obtain the weights on the values

    query: basal state z
    key: learnable values of k programs for each drug
    value: the same dim as key but represent latent space for perturbation
    """
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size, _, input_size = query.size(0), query.size(2), value.size(1)

        score = torch.bmm(query, key.transpose(1, 2))
        attn = F.softmax(score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        context = torch.bmm(attn, value)

        return context, attn

class MarkerWeight(torch.nn.Module):
    def __init__(self,hidden_dim=20,drug_dim=2,prog_dim=10):
        super(MarkerWeight, self).__init__()
        self.hidden_dim = hidden_dim
        self.drug_dim = drug_dim
        self.n_prog = prog_dim
        self.weight = torch.nn.Parameter(torch.rand(self.drug_dim,self.n_prog,self.hidden_dim))
        
    def forward(self, x):
        y = torch.matmul(x,self.weight.reshape((self.drug_dim,self.n_prog*self.hidden_dim)))
        return y.reshape((x.size(0),self.n_prog,self.hidden_dim))
    
    def get_parameters(self):
        parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
        return parameter_list

class DrugEncoder(torch.nn.Module):
    def __init__(self,hidden_dim=20,drug_dim=2,prog_dim=10):
        super(DrugEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.drug_dim = drug_dim
        self.prog_dim=prog_dim
        self.drug_weights = MarkerWeight(input_dim=self.hidden_dim,
                                         drug_dim=self.drug_dim,
                                         prog_dim=self.prog_dim)
        self.apply(init_weights)
        
    def forward(self, y):
        d = self.drug_weights(y)
        return d

class DonorWeight(torch.nn.Module):
    def __init__(self,hidden_dim=20,donor_dim=2):
        super(DonorWeight, self).__init__()
        self.hidden_dim = hidden_dim
        self.donor_dim = donor_dim
        self.weight = torch.nn.Parameter(torch.ones(self.donor_dim,self.hidden_dim))
        
    def forward(self, x):
        y = torch.matmul(x,self.weight)
        return y
    
    def get_parameters(self):
        parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
        return parameter_list

class DonorEncoder(torch.nn.Module):
    def __init__(self,input_dim=30,drug_dim=2):
        super(DonorEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.donor_dim = donor_dim
        self.donor_weights = DonorWeight(input_dim=self.hidden_dim,
                                         drug_dim=self.donor_dim)
        self.apply(init_weights)
        
    def forward(self, y):
        d = self.donor_weights(y)
        return d

#VAE model
class LINEARVAE(BaseModuleClass):
    """
    Variational auto-encoder model.

    This is an implementation of the scVI model described in [Lopez18]_

    Parameters
    ----------
    n_input
        Number of input genes
    n_batch
        Number of batches, if 0, no batch correction is performed.
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    n_continuous_cov
        Number of continuous covarites
    n_cats_per_cov
        Number of categories for each extra categorical covariate
    dropout_rate
        Dropout rate for neural networks
    dispersion
        One of the following

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    gene_likelihood
        One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    latent_distribution
        One of

        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)
    encode_covariates
        Whether to concatenate covariates to expression in encoder
    deeply_inject_covariates
        Whether to concatenate covariates into output of hidden layers in encoder/decoder. This option
        only applies when `n_layers` > 1. The covariates are concatenated to the input of subsequent hidden layers.
    use_layer_norm
        Whether to use layer norm in layers
    use_size_factor_key
        Use size_factor AnnDataField defined by the user as scaling factor in mean of conditional distribution.
        Takes priority over `use_observed_lib_size`.
    use_observed_lib_size
        Use observed library size for RNA as scaling factor in mean of conditional distribution
    library_log_means
        1 x n_batch array of means of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    library_log_vars
        1 x n_batch array of variances of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    var_activation
        Callable used to ensure positivity of the variational distributions' variance.
        When `None`, defaults to `torch.exp`.
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        n_drug: int = 3,
        n_target: int = 5,
        n_control: int = 3,
        n_prog: int=5,
        n_donor: int = 5,
        n_continuous_cov: int = 0,
        n_layers_encoder: int = 1,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        log_variational: bool = True,
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: str = "normal",
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_size_factor_key: bool = False,
        use_observed_lib_size: bool = True,
        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
        var_activation: Optional[Callable] = None,
        bias: bool = False,
    ):
        super().__init__()
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.n_drug = n_drug
        self.n_control = n_control
        self.n_prog = n_prog
        self.n_donor = n_donor
        self.n_target = n_target
        self.use_size_factor_key = use_size_factor_key
        self.use_observed_lib_size = use_size_factor_key or use_observed_lib_size
        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_means is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )

            self.register_buffer(
                "library_log_means", torch.from_numpy(library_log_means).float()
            )
            self.register_buffer(
                "library_log_vars", torch.from_numpy(library_log_vars).float()
            )

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError(
                "dispersion must be one of ['gene', 'gene-batch',"
                " 'gene-label', 'gene-cell'], but input was "
                "{}.format(self.dispersion)"
            )

        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_layers=n_layers_encoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=True,
            use_layer_norm=False,
        )
        
        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = Encoder(
            n_input,
            1,
            n_layers=1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=True,
            use_layer_norm=False,
        )
        
        self.d_encoder = DrugEncoder(n_latent,n_drug,n_prog)
        self.c_encoder = DrugEncoder(n_latent,n_control,n_prog)
        self.d_encoder_key = DrugEncoder(n_latent,n_drug,n_prog)
        self.c_encoder_key = DrugEncoder(n_latent,n_control,n_prog)
        self.donor_encoder = DonorEncoder(n_latent,n_donor)
        
        self.attention = DotProductAttention()
        
        # linear decoder goes from n_latent-dimensional space to n_input-d data
        self.decoder = LinearDecoderSCVI(
            n_latent,
            n_input,
            n_cat_list=[n_batch],
            use_batch_norm=use_batch_norm,
            use_layer_norm=False,
            bias=bias,
        )
        
        self.classifier = AdvNet(
            in_feature=n_latent,
            hidden_size=64,
            out_dim=self.n_target,
            )

    def _get_inference_input(self, tensors):
        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        pert_index = tensors['PERT_KEY']
        d = tensors['COND_KEY']
        c = tensors['CONT_KEY']
        donor = tensors['DONOR_KEY']
        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None
        
        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None
        
        input_dict = dict(
            x=x, batch_index=batch_index, pert_index=pert_index, d=d, c=c, 
            donor=donor, cont_covs=cont_covs, cat_covs=cat_covs
        )
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        Zp = inference_outputs["Zp"]
        Zc = inference_outputs["Zc"]
        Zd = inference_outputs["Zd"]
        library = inference_outputs["library"]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        
        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        size_factor_key = REGISTRY_KEYS.SIZE_FACTOR_KEY
        size_factor = (
            torch.log(tensors[size_factor_key])
            if size_factor_key in tensors.keys()
            else None
        )
        
        input_dict = dict(
            z=z,
            Zp=Zp,
            Zc=Zc,
            Zd=Zd,
            library=library,
            batch_index=batch_index,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
            size_factor=size_factor,
        )
        return input_dict
    
    def _compute_local_library_params(self, batch_index):
        """
        Computes local library parameters.

        Compute two tensors of shape (batch_index.shape[0], 1) where each
        element corresponds to the mean and variances, respectively, of the
        log library sizes in the batch the cell corresponds to.
        """
        n_batch = self.library_log_means.shape[1]
        local_library_log_means = F.linear(
            one_hot(batch_index, n_batch), self.library_log_means
        )
        local_library_log_vars = F.linear(
            one_hot(batch_index, n_batch), self.library_log_vars
        )
        return local_library_log_means, local_library_log_vars

    @auto_move_data
    def inference(self, x, d, c, donor, batch_index, pert_index, cont_covs=None, cat_covs=None, n_samples=1):
        """
        High level inference method.

        Runs the inference (encoder) model.
        """
        x_ = x
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)
        encoder_input = x_
        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()
        
        qz_m, qz_v, z = self.z_encoder(encoder_input, batch_index, *categorical_input)
        ql_m, ql_v, library_encoded = self.l_encoder(
            encoder_input, batch_index, *categorical_input
        )
        
        Zd = self.donor_encoder(donor)
        Zp = self.d_encoder(d)
        Zc = self.c_encoder(c)
        Zp_key = self.d_encoder_key(d)
        Zc_key = self.c_encoder_key(c)
        
        Zp, attP = self.attention(z.unsqueeze(1),Zp_key,Zp)
        Zp = Zp.squeeze(1)
        attP = attP.squeeze(1)
        Zc, attC = self.attention(z.unsqueeze(1),Zc_key,Zc)
        Zc = Zc.squeeze(1)
        attC = attC.squeeze(1)
        
        z = F.normalize(z, p=2, dim=1)
        prob = self.classifier(z)
        Zp = F.normalize(Zp, p=2, dim=1)
        Zc = F.normalize(Zc, p=2, dim=1)
        Zd = F.normalize(Zd, p=2, dim=1)
        
        if not self.use_observed_lib_size:
            library = library_encoded

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            # when z is normal, untran_z == z
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.z_encoder.z_transformation(untran_z)
            ql_m = ql_m.unsqueeze(0).expand((n_samples, ql_m.size(0), ql_m.size(1)))
            ql_v = ql_v.unsqueeze(0).expand((n_samples, ql_v.size(0), ql_v.size(1)))
            if self.use_observed_lib_size:
                library = library.unsqueeze(0).expand(
                    (n_samples, library.size(0), library.size(1))
                )
            else:
                library = Normal(ql_m, ql_v.sqrt()).sample()
        outputs = dict(z=z, qz_m=qz_m, qz_v=qz_v, ql_m=ql_m, ql_v=ql_v,
                       prob=prob, Zp=Zp, Zc=Zc, Zd=Zd, library=library, 
                       attP=attP, attC=attC)
        return outputs


    @auto_move_data
    def generative(self,z,Zp,Zc,Zd,library,batch_index,cont_covs=None,cat_covs=None,size_factor=None,y=None,transform_batch=None):#
        """Runs the generative model."""
        # Likelihood distribution
        zA = z+Zp+Zc+Zd
        decoder_input = zA
        
        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        if not self.use_size_factor_key:
            size_factor = library

        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion,
            decoder_input,
            size_factor,
            batch_index,
            *categorical_input,
        )
        if self.dispersion == "gene-label":
            px_r = F.linear(
                one_hot(y, self.n_labels), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r

        px_r = torch.exp(px_r)

        if self.gene_likelihood == "zinb":
            px = ZeroInflatedNegativeBinomial(
                mu=px_rate,
                theta=px_r,
                zi_logits=px_dropout,
                scale=px_scale,
            )
        elif self.gene_likelihood == "nb":
            px = NegativeBinomial(mu=px_rate, theta=px_r)
        elif self.gene_likelihood == "poisson":
            px = Poisson(px_rate)

        # Priors
        if self.use_observed_lib_size:
            pl = None
        else:
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)
            pl = Normal(local_library_log_means, local_library_log_vars.sqrt())
        pz = Normal(torch.zeros_like(zA), torch.ones_like(zA))
        return dict(
            zA=zA,
            px=px,
            pl=pl,
            pz=pz
            )

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ):
        x = tensors[REGISTRY_KEYS.X_KEY]
        l = tensors["TARGET_KEY"]
        
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        ql_m = inference_outputs["ql_m"]
        ql_v = inference_outputs["ql_v"]
        
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)

        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(
            dim=1
        )

        if not self.use_observed_lib_size:
            kl_divergence_l = kl(
                Normal(ql_m, torch.sqrt(ql_v)),
                generative_outputs["pl"],
                ).sum(dim=1)
        else:
            kl_divergence_l = 0.0
        
        reconst_loss = -generative_outputs["px"].log_prob(x).sum(-1)
        
        advers_loss = -torch.nn.BCELoss(reduction='sum')(inference_outputs["prob"],l)
        #prob_target = inference_outputs["prob"][l_==0,:]
        #prob_source = inference_outputs["prob"][l_==1,:]
        #advers_loss = -(prob_source.mean(0) - prob_target.mean(0)).mean()
        
        #ent_penalty = entropy(generative_outputs["zA"])
        #att_penalty = entropy(inference_outputs["attP"],temp=0.5)+entropy(inference_outputs["attC"],temp=0.5)
        
        kl_local_for_warmup = kl_divergence_z
        kl_local_no_warmup = kl_divergence_l

        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

        #loss = torch.mean(reconst_loss*0.5 + weighted_kl_local + ent_penalty*0.25 + advers_loss)
        #loss = torch.mean(reconst_loss*0.5 + weighted_kl_local + ent_penalty*0.2 + advers_loss)
        loss = torch.mean(reconst_loss*0.5 + weighted_kl_local + advers_loss)
        #loss = torch.mean(reconst_loss*0.25 + weighted_kl_local*0.5 + ent_penalty*0.1 + advers_loss + att_penalty*0.1)
        
        kl_local = dict(
            kl_divergence_l=kl_divergence_l, kl_divergence_z=kl_divergence_z
        )
        kl_global = torch.tensor(0.0)
        return LossRecorder(loss, reconst_loss, kl_local, kl_global)


    @torch.no_grad()
    def sample(
        self,
        tensors,
        n_samples=1,
        library_size=1,
    ) -> np.ndarray:
        r"""
        Generate observation samples from the posterior predictive distribution.

        The posterior predictive distribution is written as :math:`p(\hat{x} \mid x)`.

        Parameters
        ----------
        tensors
            Tensors dict
        n_samples
            Number of required samples for each cell
        library_size
            Library size to scale scamples to

        Returns
        -------
        x_new : :py:class:`torch.Tensor`
            tensor with shape (n_cells, n_genes, n_samples)
        """
        inference_kwargs = dict(n_samples=n_samples)
        _, generative_outputs, = self.forward(
            tensors,
            inference_kwargs=inference_kwargs,
            compute_loss=False,
        )

        dist = generative_outputs["px"]
        if self.gene_likelihood == "poisson":
            l_train = generative_outputs["px"].mu
            l_train = torch.clamp(l_train, max=1e8)
            dist = torch.distributions.Poisson(
                l_train
            )  # Shape : (n_samples, n_cells_batch, n_genes)
        if n_samples > 1:
            exprs = dist.sample().permute(
                [1, 2, 0]
            )  # Shape : (n_cells_batch, n_genes, n_samples)
        else:
            exprs = dist.sample()

        return exprs.cpu()


    @torch.no_grad()
    @auto_move_data
    def marginal_ll(self, tensors, n_mc_samples):
        sample_batch = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        to_sum = torch.zeros(sample_batch.size()[0], n_mc_samples)

        for i in range(n_mc_samples):
            # Distribution parameters and sampled variables
            inference_outputs, _, losses = self.forward(tensors)
            qz = inference_outputs["qz"]
            ql = inference_outputs["ql"]
            z = inference_outputs["z"]
            library = inference_outputs["library"]

            # Reconstruction Loss
            reconst_loss = losses.reconstruction_loss
            # Log-probabilities
            p_z = (
                Normal(torch.zeros_like(qz.loc), torch.ones_like(qz.scale))
                .log_prob(z)
                .sum(dim=-1)
            )
            p_x_zl = -reconst_loss
            q_z_x = qz.log_prob(z).sum(dim=-1)
            log_prob_sum = p_z + p_x_zl - q_z_x

            if not self.use_observed_lib_size:
                (
                    local_library_log_means,
                    local_library_log_vars,
                ) = self._compute_local_library_params(batch_index)

                p_l = (
                    Normal(local_library_log_means, local_library_log_vars.sqrt())
                    .log_prob(library)
                    .sum(dim=-1)
                )
                q_l_x = ql.log_prob(library).sum(dim=-1)

                log_prob_sum += p_l - q_l_x

            to_sum[:, i] = log_prob_sum

        batch_log_lkl = logsumexp(to_sum, dim=-1) - np.log(n_mc_samples)
        log_lkl = torch.sum(batch_log_lkl).item()
        return log_lkl

class VariationalCPA(RNASeqMixin, VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """
    Linearly-decoded VAE [Svensson20]_.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :func:`~scvi.data.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder NN.
    dropout_rate
        Dropout rate for neural networks.
    dispersion
        One of the following:

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    gene_likelihood
        One of:

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    latent_distribution
        One of:

        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal distribution (Normal(0, I) transformed by softmax)
    **model_kwargs
        Keyword args for :class:`~scvi.module.LDVAE`

    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> scvi.data.setup_anndata(adata, batch_key="batch")
    >>> vae = scvi.model.LinearSCVI(adata)
    >>> vae.train()
    >>> adata.var["loadings"] = vae.get_loadings()

    Notes
    -----
    See further usage examples in the following tutorials:

    1. :doc:`/user_guide/notebooks/linear_decoder`
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        n_batch: int = 0,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "nb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        log_variational: bool = True,
        use_batch_norm: bool = True,
        bias: bool = False,
        **model_kwargs,
    ):
        super(VariationalCPA, self).__init__(adata)
        self.module = LINEARVAE(
            n_input=self.summary_stats["n_vars"],
            n_batch=self.summary_stats["n_batch"],
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers_encoder=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            **model_kwargs,
        )
        self._model_summary_string = (
            "VariationalCPA Model with the following params: \nn_hidden: {}, n_latent: {}, n_layers: {}, dropout_rate: "
            "{}, dispersion: {}, gene_likelihood: {}, latent_distribution: {}"
        ).format(
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
            dispersion,
            gene_likelihood,
            latent_distribution,
        )
        
        self.use_batch_norm = use_batch_norm
        self.n_latent = n_latent
        self.init_params_ = self._get_init_params(locals())
        
    @torch.no_grad()
    def get_loadings(self) -> np.ndarray:
        """Extract per-gene weights (for each Z, shape is genes by dim(Z)) in the linear decoder."""
        # This is BW, where B is diag(b) batch norm, W is weight matrix
        if self.use_batch_norm is True:
            w = self.module.decoder.factor_regressor.fc_layers[0][0].weight
            bn = self.module.decoder.factor_regressor.fc_layers[0][1]
            sigma = torch.sqrt(bn.running_var + bn.eps)
            gamma = bn.weight
            b = gamma / sigma
            b_identity = torch.diag(b)
            loadings = torch.matmul(b_identity, w)
        else:
            loadings = self.module.decoder.factor_regressor.fc_layers[0][0].weight
        loadings = loadings.detach().cpu().numpy()
        if self.module.n_batch > 1:
            loadings = loadings[:, : -self.module.n_batch]

        return loadings
    
    def get_pert_loadings(self) -> pd.DataFrame:#TO DO
        """
        Extract per-gene weights in the linear decoder.

        Shape is genes by `n_latent`.

        """
        weights=[]
        for p in self.module.d_encoder.drug_weights.parameters():
            weights.append(p)
        w = weights[0]
        #w = F.normalize(w, p=2, dim=2)
        loadings=torch.Tensor.cpu(w).detach().numpy()
        #loadings = pd.DataFrame(loadings.T)

        return loadings
    
    def get_donor_loadings(self) -> pd.DataFrame:#TO DO
        """
        Extract per-gene weights in the linear decoder.

        Shape is genes by `n_latent`.

        """
        weights=[]
        for p in self.module.donor_encoder.drug_weights.parameters():
            weights.append(p)
        w = weights[0]
        #w = F.normalize(w, p=2, dim=1)
        loadings=torch.Tensor.cpu(w).detach().numpy()
        loadings = pd.DataFrame(loadings.T)

        return loadings
    
    @torch.no_grad()
    def get_latent_embedding(
        self,
        adata: Optional[AnnData] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Returns the latent library size for each cell.

        This is denoted as :math:`\ell_n` in the totalVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        give_mean
            Return the mean or a sample from the posterior distribution.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        """
        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")

        adata = self._validate_anndata(adata)
        post = self._make_data_loader(
            adata=adata, batch_size=batch_size
        )#indices=indices, 
        embedding = []
        for tensors in post:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            z = outputs["z"]
            embedding += [z.cpu()]
        return np.array(torch.cat(embedding))
    
    @torch.no_grad()
    def get_pert_embedding(
        self,
        adata: Optional[AnnData] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Returns the latent library size for each cell.

        This is denoted as :math:`\ell_n` in the totalVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        give_mean
            Return the mean or a sample from the posterior distribution.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        """
        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")

        adata = self._validate_anndata(adata)
        post = self._make_data_loader(
            adata=adata, batch_size=batch_size
        )#indices=indices, 
        embedding = []
        atts = []
        for tensors in post:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            Zp = outputs["Zp"]
            embedding += [Zp.cpu()]
            
            attP = outputs["attP"]
            atts += [attP.cpu()]
            
        return np.array(torch.cat(embedding)),np.array(torch.cat(atts))
    
    @torch.no_grad()
    def get_control_embedding(
        self,
        adata: Optional[AnnData] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Returns the latent library size for each cell.

        This is denoted as :math:`\ell_n` in the totalVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        give_mean
            Return the mean or a sample from the posterior distribution.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        """
        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")

        adata = self._validate_anndata(adata)
        post = self._make_data_loader(
            adata=adata, batch_size=batch_size
        )#indices=indices, 
        embedding = []
        atts = []
        for tensors in post:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            Zp = outputs["Zc"]
            embedding += [Zp.cpu()]
            
            attC = outputs["attC"]
            atts += [attC.cpu()]
            
        return np.array(torch.cat(embedding)),np.array(torch.cat(atts))
    
    @torch.no_grad()
    def get_donor_embedding(
        self,
        adata: Optional[AnnData] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Returns the latent library size for each cell.

        This is denoted as :math:`\ell_n` in the totalVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        give_mean
            Return the mean or a sample from the posterior distribution.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        """
        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")

        adata = self._validate_anndata(adata)
        post = self._make_data_loader(
            adata=adata, batch_size=batch_size
        )#indices=indices, 
        embedding = []
        for tensors in post:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            z = outputs["Zd"]
            embedding += [z.cpu()]
        return np.array(torch.cat(embedding))
    
    @torch.no_grad()
    def get_embedding(
        self,
        adata: Optional[AnnData] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Returns the latent library size for each cell.

        This is denoted as :math:`\ell_n` in the totalVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        give_mean
            Return the mean or a sample from the posterior distribution.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        """
        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")

        adata = self._validate_anndata(adata)
        post = self._make_data_loader(
            adata=adata, batch_size=batch_size
        )#indices=indices, 
        embedding = []
        for tensors in post:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            Za = outputs['z']+outputs["Zp"]
            embedding += [Za.cpu()]
        return np.array(torch.cat(embedding))
    
    @torch.no_grad()
    def predict(
        self,
        adata: Optional[AnnData] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Returns the latent library size for each cell.

        This is denoted as :math:`\ell_n` in the totalVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        give_mean
            Return the mean or a sample from the posterior distribution.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        """
        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")

        adata = self._validate_anndata(adata)
        post = self._make_data_loader(
            adata=adata, batch_size=batch_size
        )#indices=indices, 
        preditions = []
        for tensors in post:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            generative_inputs = self.module._get_generative_input(tensors,outputs)
            outputs = self.module.generative(**generative_inputs)
            out = outputs['px'].mu
            preditions += [out.cpu()]
        return np.array(torch.cat(preditions))
    
    # TO DO
    def train(
        self,
        max_epochs: int = 500,
        lr: float = 1e-4,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        weight_decay: float = 1e-3,
        eps: float = 1e-08,
        early_stopping: bool = True,
        save_best: bool = True,
        check_val_every_n_epoch: Optional[int] = None,
        n_steps_kl_warmup: Optional[int] = None,
        n_epochs_kl_warmup: Optional[int] = 50,
        plan_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Trains the model using amortized variational inference.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset.
        lr
            Learning rate for optimization.
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        weight_decay
            weight decay regularization term for optimization
        eps
            Optimizer eps
        early_stopping
            Whether to perform early stopping with respect to the validation set.
        save_best
            Save the best model state with respect to the validation loss, or use the final
            state in the training procedure
        check_val_every_n_epoch
            Check val every n train epochs. By default, val is not checked, unless `early_stopping` is `True`.
            If so, val is checked every epoch.
        n_steps_kl_warmup
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
            Only activated when `n_epochs_kl_warmup` is set to None. If `None`, defaults
            to `floor(0.75 * adata.n_obs)`.
        n_epochs_kl_warmup
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`.
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        update_dict = dict(
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            n_steps_kl_warmup=n_steps_kl_warmup,
            #optimizer="Adam",
            optimizer="AdamW",
            )
        if plan_kwargs is not None:
            plan_kwargs.update(update_dict)
        else:
            plan_kwargs = update_dict

        if save_best:
            if "callbacks" not in kwargs.keys():
                kwargs["callbacks"] = []
            kwargs["callbacks"].append(
                SaveBestState(monitor="reconstruction_loss_validation")
            )

        data_splitter = DataSplitter(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )

        training_plan = TrainingPlan(self.module, **plan_kwargs)
        
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            early_stopping=early_stopping,
            check_val_every_n_epoch=check_val_every_n_epoch,
            early_stopping_monitor="reconstruction_loss_validation",
            early_stopping_patience=50,
            **kwargs,
        )
        return runner()
    
    # TO DO
    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        labels_key: Optional[str] = None,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        pert_key: Optional[str] = None,
        cond_key: Optional[str] = None,
        cont_key: Optional[str] = None,
        target_key: Optional[str] = None,
        donor_key: Optional[str] = None,
        size_factor_key: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        %(summary)s.

        Parameters
        ----------
        %(param_layer)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            CategoricalObsField(registry_key="PERT_KEY", obs_key=pert_key),
            NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False
            ),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
            ObsmField(
                registry_key="COND_KEY", obsm_key=cond_key
            ),
            ObsmField(
                registry_key="CONT_KEY", obsm_key=cont_key
            ),
            ObsmField(
                registry_key="TARGET_KEY", obsm_key=target_key
            ),
            ObsmField(
                registry_key="DONOR_KEY", obsm_key=donor_key
            ),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
