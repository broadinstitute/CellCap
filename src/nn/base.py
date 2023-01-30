import logging
import numpy as np

import torch
from torch import logsumexp
import torch.nn.functional as F
from torch.distributions import Normal, Poisson
from torch.distributions import kl_divergence as kl

from scvi import REGISTRY_KEYS
from scvi._compat import Literal
from scvi.nn import Encoder, LinearDecoderSCVI, one_hot
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.distributions import ZeroInflatedNegativeBinomial, NegativeBinomial

logger = logging.getLogger(__name__)

from typing import Callable, Iterable, Optional

torch.backends.cudnn.benchmark = True

from .advclassifier import AdvNet
from .attention import DotProductAttention
from .drugencoder import DrugEncoder
from .donorencoder import DonorEncoder

##-------------------------------------
def entropy(x,temp=1.0):
    p = F.softmax(x/temp, dim=1)# + 1e-8
    logp = F.log_softmax(x/temp,dim=1)# + 1e-8
    return -(p*logp).sum(dim=1)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def cal_off_diagonal_corr(z):
    c = z[0,:,:].T @ z[0,:,:]
    off_diag = off_diagonal(c).pow_(2).sum()
    for i in range(1,z.shape[0]):
        c = z[i,:,:].T @ z[i,:,:]
        off_diag += off_diagonal(c).pow_(2).sum()
    return off_diag

class LINEARVAE(BaseModuleClass):

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_drug: int = 3,
        n_target: int = 5,
        n_control: int = 3,
        n_prog: int=5,
        n_donor: int = 5,
        n_layers_encoder: int = 1,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        log_variational: bool = True,
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: str = "normal",
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_size_factor_key: bool = False,
        use_observed_lib_size: bool = True,
        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
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

        self.d_encoder = DrugEncoder(n_latent,n_drug,n_prog,key=True)
        self.c_encoder = DrugEncoder(n_latent,n_control,n_prog,key=True)
        self.d_encoder_key = DrugEncoder(n_latent,n_drug,n_prog,key=True)
        self.c_encoder_key = DrugEncoder(n_latent,n_control,n_prog,key=True)

        self.donor_encoder = DonorEncoder(n_latent,n_donor)

        self.ard_d = ARDregularizer(n_drug,n_prog)
        self.ard_c = ARDregularizer(n_control,n_prog)

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
    def inference(self, x, d, c, donor, batch_index, pert_index,
                  cont_covs=None, cat_covs=None, n_samples=1):
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
        Zp_key = self.d_encoder_key(d)
        Zc = self.c_encoder(c)
        Zc_key = self.c_encoder_key(c)

        Zp, attP = self.attention(z.unsqueeze(1),Zp_key,Zp)
        Zp = Zp.squeeze(1)
        attP = attP.squeeze(1)
        Zc, attC = self.attention(z.unsqueeze(1),Zc_key,Zc)
        Zc = Zc.squeeze(1)
        attC = attC.squeeze(1)

        alpha_ip_d = self.ard_d(d)
        alpha_ip_c = self.ard_c(c)

        prob = self.classifier(z)

        z = F.normalize(z, p=2, dim=1)
        Zp = F.normalize(Zp, p=2, dim=1)
        Zc = F.normalize(Zc, p=2, dim=1)
        Zd = F.normalize(Zd, p=2, dim=1)

        if not self.use_observed_lib_size:
            library = library_encoded

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
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
                       attP=attP, attC=attC,
                       alpha_ip_d=alpha_ip_d,alpha_ip_c=alpha_ip_c,
                       )
        return outputs

    @auto_move_data
    def generative(self,z,Zp,Zc,Zd,library,batch_index,cont_covs=None,cat_covs=None,
                   size_factor=None,y=None,transform_batch=None):
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

        kl_local_for_warmup = kl_divergence_z
        kl_local_no_warmup = kl_divergence_l
        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

        advers_loss = torch.nn.BCELoss(reduction='sum')(inference_outputs["prob"],l)

        ent_penalty = entropy(generative_outputs["zA"])
        off_penalty = cal_off_diagonal_corr(self.d_encoder.drug_weights.weight)

        ard_reg_d = Normal(loc=0., scale=1. / inference_outputs["alpha_ip_d"]).log_prob(inference_outputs["attP"]).sum()
        ard_reg_c = Normal(loc=0., scale=1. / inference_outputs["alpha_ip_c"]).log_prob(inference_outputs["attC"]).sum()
        ard_reg = ard_reg_d + ard_reg_c

        loss = torch.mean(reconst_loss*0.5 + weighted_kl_local + advers_loss + ent_penalty*0.2 + ard_reg*0.001) + off_penalty*0.1

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
