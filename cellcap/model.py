"""The CellCap model"""

import logging

import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence as kl
from torch.distributions import Normal, Poisson, Laplace

from scvi import REGISTRY_KEYS
from scvi.nn import Encoder, one_hot
from scvi.distributions import NegativeBinomial
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data

from typing import Dict, Literal
from .mixins import CellCapMixin
from .nn.advclassifier import AdvNet
from .nn.decoder import LinearDecoderSCVI

torch.backends.cudnn.benchmark = True
logger = logging.getLogger(__name__)


class CellCapModel(BaseModuleClass, CellCapMixin):
    def __init__(
        self,
        n_input: int,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_drug: int = 3,
        n_prog: int = 5,
        n_covar: int = 5,
        n_layers_encoder: int = 1,
        n_head: int = 2,
        dropout_rate: float = 0.25,
        lamda: float = 1.0,
        kl_weight: float = 1.0,
        rec_weight: float = 2.0,
        ard_kl_weight: float = 0.2,
        dispersion: str = "gene",
        log_variational: bool = True,
        gene_likelihood: Literal["nb", "poisson"] = "nb",
        latent_distribution: str = "normal",
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        bias: bool = False,
    ):
        super().__init__()
        self.n_input = n_input
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        self.n_labels = n_labels
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.n_drug = n_drug
        self.n_prog = n_prog
        self.n_covar = n_covar
        self.n_head = n_head

        # hyperparameter used in training
        self.lamda = lamda
        self.kl_weight = kl_weight
        self.rec_weight = rec_weight
        self.ard_kl_weight = ard_kl_weight

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError(
                "dispersion must be one of ['gene', "
                " 'gene-label', 'gene-cell'], but input was "
                "{}".format(self.dispersion)
            )

        # p stands for perturbation
        # k stands for the dimension of latent space
        # d stands for the number of categorical covariates
        # q stands for the dimension of transcriptional response

        self.alpha_q = torch.nn.Parameter(torch.zeros(n_prog))
        self.H_pq = torch.nn.Parameter(torch.zeros(n_drug, n_prog))

        w_qk = torch.empty(n_prog, n_latent)
        self.w_qk = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(
                w_qk, gain=torch.nn.init.calculate_gain("relu")
            )
        )
        w_covar_dk = torch.empty(n_covar, n_latent)
        self.w_covar_dk = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(
                w_covar_dk, gain=torch.nn.init.calculate_gain("relu")
            )
        )
        H_key = torch.empty(n_drug, n_prog, n_latent, n_head)
        self.H_key = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(
                H_key, gain=torch.nn.init.calculate_gain("relu")
            )
        )

        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_layers=n_layers_encoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm in ["encoder", "both"],
            use_layer_norm=False,
        )

        # linear decoder goes from n_latent-dimensional space to n_input-d data
        self.decoder = LinearDecoderSCVI(
            n_latent,
            n_input,
            use_batch_norm=use_batch_norm in ["decoder", "both"],
            use_layer_norm=False,
            bias=bias,
        )

        self.discriminator = AdvNet(
            in_feature=self.n_latent, hidden_size=128, out_dim=self.n_drug
        )

    def _get_inference_input(self, tensors):
        x = tensors[REGISTRY_KEYS.X_KEY]
        p = tensors["TARGET_KEY"]
        covar = tensors["COVAR_KEY"]

        input_dict = dict(
            x=x,
            p=p,
            covar=covar,
        )
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        library = inference_outputs["library"]
        z_basal = inference_outputs["z_basal"]
        delta_z = inference_outputs["delta_z"]
        delta_z_covar = inference_outputs["delta_z_covar"]

        input_dict = dict(
            z_basal=z_basal,
            delta_z=delta_z,
            delta_z_covar=delta_z_covar,
            library=library,
        )
        return input_dict

    @auto_move_data
    def inference(self, x, p, covar, n_samples=1):
        """
        High level inference method.

        Runs the inference (encoder) model.
        """
        x_ = x
        library = torch.log(x.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)
        encoder_input = x_

        h = torch.matmul(p, F.softplus(self.H_pq))

        qz_m, qz_v, z_basal = self.z_encoder(encoder_input)
        delta_z_covar = torch.matmul(covar, self.w_covar_dk)

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z_basal = self.z_encoder.z_transformation(untran_z)
            library = library.unsqueeze(0).expand(
                (n_samples, library.size(0), library.size(1))
            )

        # Attention
        attn = []
        for i in range(self.n_head):
            key = torch.matmul(
                p,
                self.H_key[:, :, :, i].reshape(
                    (self.n_drug, self.n_prog * self.n_latent)
                ),
            )
            key = key.reshape((p.size(0), self.n_prog, self.n_latent))
            score = torch.bmm(z_basal.unsqueeze(1), key.transpose(1, 2))
            score = score.view(-1, self.n_prog)
            attn += [F.softmax(score, dim=1)]
        attn = torch.max(torch.stack(attn, dim=2), 2)[0]
        H_attn = attn * h

        prob = self.discriminator(z_basal)
        delta_z = torch.matmul(H_attn, self.w_qk.tanh())

        outputs = dict(
            z_basal=z_basal,
            qz_m=qz_m,
            qz_v=qz_v,
            delta_z=delta_z,
            delta_z_covar=delta_z_covar,
            library=library,
            h=h,
            prob=prob,
            attn=attn,
            H_attn=H_attn,
        )

        return outputs

    @auto_move_data
    def generative(
        self, z_basal, delta_z, delta_z_covar, library, y=None, transform_batch=None
    ):
        """Runs the generative model."""
        # Likelihood distribution

        z = z_basal + delta_z + delta_z_covar
        decoder_input = z

        categorical_input = tuple()
        size_factor = library

        px_scale, px_r, px_rate = self.decoder(
            self.dispersion,
            decoder_input,
            size_factor,
            *categorical_input,
        )
        if self.dispersion == "gene-label":
            # px_r gets transposed - last dimension is nb genes
            px_r = F.linear(one_hot(y, self.n_labels), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r

        px_r = torch.exp(px_r)

        if self.gene_likelihood == "nb":
            px = NegativeBinomial(mu=px_rate, theta=px_r)
        elif self.gene_likelihood == "poisson":
            px = Poisson(px_rate)

        # Priors
        pl = None
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))
        return dict(z=z, px=px, pl=pl, pz=pz)

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
    ):
        x = tensors[REGISTRY_KEYS.X_KEY]
        p = tensors["TARGET_KEY"]

        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]

        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)

        H_attn = inference_outputs["H_attn"][p.sum(1) > 0, :]
        laploc = torch.zeros_like(self.alpha_q)

        # reconstruction loss
        rec_loss = -generative_outputs["px"].log_prob(x).sum(-1) * self.rec_weight
        
        # KL divergence
        kl_divergence_z = kl(
            Normal(qz_m, torch.sqrt(qz_v)), 
            Normal(mean, scale)).sum(
            dim=1
        )
        
        kl_divergence_ard = -1 * (
            Laplace(loc=laploc, scale=self.alpha_q.sigmoid())
            .log_prob(H_attn)
            .sum(-1)
        ) * self.ard_kl_weight
        
        weighted_kl_local = (
            self.kl_weight * kl_divergence_z
        )
        
        adv_loss = torch.nn.BCELoss(reduction='mean')(
            inference_outputs["prob"],p
        ) * self.lamda
        scale_factor = ((torch.mean(rec_loss)/adv_loss)*0.05).detach()
        
        loss = (
            torch.mean(
                rec_loss + weighted_kl_local
            )
            + torch.mean(kl_divergence_ard)
            + adv_loss * scale_factor
        )

        kl_local = dict(
            kl_divergence_z=kl_divergence_z,
        )

        # extra metrics for logging
        extra_metrics = {
            "adv_loss": adv_loss,
        }

        return LossOutput(
            loss=loss,
            kl_local=kl_local,
            extra_metrics=extra_metrics,
            reconstruction_loss=rec_loss,
        )


def logging_dict_from_tensor(x: torch.Tensor, name: str) -> Dict[str, torch.Tensor]:
    """Take a > zero-dimensional tensor and flatten, returning a dictionary
    with each value being a 0-d tensor, appropriate for logging."""
    flat_x = x.flatten()
    return {f"{name}_{i}": flat_x[i] for i in range(len(flat_x))}
