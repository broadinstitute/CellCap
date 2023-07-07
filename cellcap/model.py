"""The CellCap model"""

import logging

import torch
import torch.nn.functional as F
from torch.distributions import Normal, Poisson, Laplace, Distribution, HalfCauchy
from torch.distributions import kl_divergence as kl
from torch.distributions.kl import register_kl
from pyro.distributions import Delta
import numpy as np

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
        n_donor: int = 5,
        n_layers_encoder: int = 1,
        n_head: int = 1,
        dropout_rate: float = 0.25,
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
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        self.n_labels = n_labels
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.n_drug = n_drug
        self.n_prog = n_prog
        self.n_donor = n_donor
        self.n_head = n_head

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
        # q stands for the dimension of transcriptional response
        # k stands for the dimension of latent space
        # d stands for the number of donors

        # self.alpha_q = torch.nn.Parameter(torch.zeros(1, n_prog))
        self.H_pq = torch.nn.Parameter(torch.ones(n_drug, n_prog))
        # self.log_alpha_pq = torch.nn.Parameter(torch.zeros(n_drug, n_prog))

        # hyperparamters for ARD
        self.b_q = torch.nn.Parameter((torch.ones(n_prog) / np.sqrt(n_prog)).logit())
        # self.b_q = torch.nn.Parameter(-2 * torch.ones(n_prog))
        self.c_pq = torch.nn.Parameter(torch.ones(n_drug, n_prog))
        self.laplace_scale = torch.nn.Parameter(-2 * torch.ones([1]))

        self.adv_penalty = 1.

        w_qk = torch.empty(n_prog, n_latent)
        self.w_qk = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(
                w_qk, gain=torch.nn.init.calculate_gain("relu")
            )
        )
        w_donor_dk = torch.empty(n_donor, n_latent)
        self.w_donor_dk = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(
                w_donor_dk, gain=torch.nn.init.calculate_gain("relu")
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
            in_feature=self.n_latent, hidden_size=128, out_dim=self.n_drug + 1  # include control class
        )

    def _get_inference_input(self, tensors):
        x = tensors[REGISTRY_KEYS.X_KEY]
        p = tensors["TARGET_KEY"]

        input_dict = dict(
            x=x,
            p=p,
        )
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        library = inference_outputs["library"]
        z_basal = inference_outputs["z_basal"]
        donor = tensors["DONOR_KEY"]
        perturbation_np = tensors["TARGET_KEY"]
        v_nq = inference_outputs["v_nq"]
        u_q = inference_outputs["u_q"]

        input_dict = dict(
            z_basal=z_basal,
            v_nq=v_nq,
            u_q=u_q,
            perturbation_np=perturbation_np,
            library=library,
            donor=donor,
        )
        return input_dict

    @auto_move_data
    def inference(self, x, p, n_samples=1):
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

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z_basal = self.z_encoder.z_transformation(untran_z)
            library = library.unsqueeze(0).expand(
                (n_samples, library.size(0), library.size(1))
            )

        # Attention
        key = torch.matmul(
            p,
            self.H_key[:, :, :, 0].reshape((self.n_drug, self.n_prog * self.n_latent)),
        )
        key = key.reshape((p.size(0), self.n_prog, self.n_latent))
        score = torch.bmm(z_basal.unsqueeze(1), key.transpose(1, 2))
        score = score.view(-1, self.n_prog)
        attn = F.softmax(score, dim=1)
        for i in range(1, self.n_head):
            key = torch.matmul(
                p,
                self.H_key[:, :, :, i].reshape(
                    (self.n_drug, self.n_prog * self.n_latent)
                ),
            )
            key = key.reshape((p.size(0), self.n_prog, self.n_latent))
            score = torch.bmm(z_basal.unsqueeze(1), key.transpose(1, 2))
            score = score.view(-1, self.n_prog)
            a = F.softmax(score, dim=1)
            attn = attn + a
        H_attn = attn * h

        # posteriors for v_nq and u_q
        v_nq_posterior_sample = H_attn
        # v_nq_posterior = Laplace(loc=H_attn, scale=self.laplace_scale.exp())
        # v_nq_posterior_sample = v_nq_posterior.sample()
        # u_q_posterior = Laplace(
        #     loc=v_nq_posterior_sample.sum(dim=0), scale=self.laplace_scale.exp()
        # )
        u_q_posterior = Delta(v_nq_posterior_sample.sum(dim=0))
        u_q_posterior_sample = u_q_posterior.sample()

        prob = self.discriminator(z_basal)

        outputs = dict(
            z_basal=z_basal,
            qz_m=qz_m,
            qz_v=qz_v,
            z_basal_nk_posterior=Normal(qz_m, qz_v),
            library=library,
            h=h,
            # v_nq_posterior=v_nq_posterior,
            u_q_posterior=u_q_posterior,
            v_nq=v_nq_posterior_sample,
            u_q=u_q_posterior_sample,
            prob=prob,
            attn=attn,
            H_attn=H_attn,
        )
        return outputs

    @auto_move_data
    def generative(
        self,
        z_basal,
        u_q,
        v_nq,
        library,
        perturbation_np,
        donor,
        y=None,
        transform_batch=None,
    ):
        """Runs the generative model."""
        # Likelihood distribution

        # priors on u_q and v_nq
        u_q_prior = HalfCauchy(scale=self.b_q.sigmoid())
        # v_nq_prior = Laplace(
        #     loc=0.0,
        #     scale=(
        #         u_q.unsqueeze(0) * torch.matmul(perturbation_np, self.c_pq)
        #     ).sigmoid() + 1e-10,
        # )

        # donor contribution
        delta_z_donor = torch.matmul(donor, self.w_donor_dk)

        # compute z perturbation
        delta_z = torch.matmul(v_nq, self.w_qk)

        # final z
        z = z_basal + delta_z + delta_z_donor
        decoder_input = z

        # for scvi
        categorical_input = tuple()
        size_factor = library

        px_scale, px_r, px_rate = self.decoder(
            self.dispersion,
            decoder_input,
            size_factor,
            *categorical_input,
        )
        if self.dispersion == "gene-label":
            px_r = F.linear(
                one_hot(y, self.n_labels), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene":
            px_r = self.px_r

        px_r = torch.exp(px_r)

        if self.gene_likelihood == "nb":
            px = NegativeBinomial(mu=px_rate, theta=px_r)
        elif self.gene_likelihood == "poisson":
            px = Poisson(px_rate)

        # Priors
        pl = None
        z_basal_nk_prior = Normal(torch.zeros_like(z), torch.ones_like(z))
        return dict(
            z=z,
            px=px,
            pl=pl,
            z_basal_nk_prior=z_basal_nk_prior,
            u_q_prior=u_q_prior,
            # v_nq_prior=v_nq_prior,
        )

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
        # h_kl_weight: float = 1.0,
        g_kl_weight: float = 0.1,
        rec_weight: float = 1.0,
        lamda: float = 1.0,
    ):
        x = tensors[REGISTRY_KEYS.X_KEY]
        perturbations = tensors["TARGET_KEY"]

        # add one-hot control column as first column of perturbations
        perturbations = torch.cat(
            [(perturbations.sum(dim=-1, keepdim=True) == 0).float(),  # control cells
             perturbations],
            dim=-1,
        )

        # reconstruction loss
        rec_loss = -generative_outputs["px"].log_prob(x).sum(-1) * rec_weight

        # KL divergence
        kl_divergence_z_n = kl(
            inference_outputs["z_basal_nk_posterior"],
            generative_outputs["z_basal_nk_prior"],
        ).sum(1)

        # KL divergence for u_q
        kl_divergence_u_q = kl(
            inference_outputs["u_q_posterior"],
            generative_outputs["u_q_prior"],
        )

        # # KL divergence for v_nq
        # kl_divergence_v_n = kl(
        #     inference_outputs["v_nq_posterior"],
        #     generative_outputs["v_nq_prior"],
        # ).sum(1)

        # # mask control cells from u_q and v_nq divergences
        # perturbed_mask = perturbations.sum(dim=-1) > 0
        # kl_divergence_v_n = torch.where(
        #     perturbed_mask, kl_divergence_v_n, torch.zeros_like(kl_divergence_v_n)
        # )

        with torch.no_grad():
            adv_threshold = 0.0  # some value we choose
            adv_decay = 0.99
            # predictions = torch.argmax(inference_outputs["prob"], dim=-1)
            # truth = perturbations
            # adv_accuracy = (torch.gather(truth, 1, predictions.unsqueeze(1)).sum().squeeze()
            #                 / (perturbations.sum(dim=-1) > 0).sum())
            # random_accuracy = 1. / perturbations.shape[1]
            # adv_penalty = F.relu(10. * (torch.exp(2. * (adv_accuracy - random_accuracy - adv_threshold)) - 1.))

            bce_loss = torch.nn.BCELoss(reduction="sum")(inference_outputs["prob"].detach(), perturbations)
            random_bce_loss = torch.nn.BCELoss(reduction="sum")(
                torch.ones_like(perturbations) * perturbations.sum(dim=0, keepdim=True) / perturbations.shape[0],
                perturbations,
            )
            adv_accuracy = F.relu(random_bce_loss - bce_loss)
            adv_penalty = 5 * adv_accuracy
            self.adv_penalty = F.relu(self.adv_penalty * adv_decay + adv_penalty * (1. - adv_decay))

        final_adv_penalty = lamda + self.adv_penalty.detach()

        # cell_classes = torch.where(
        #     perturbations.sum(dim=-1) > 0,
        #     torch.argmax(perturbations, dim=-1) + 1,
        #     torch.zeros_like(),
        # )

        adv_loss = (
            torch.nn.BCELoss(reduction="sum")(inference_outputs["prob"], perturbations)
            # torch.nn.CrossEntropyLoss(reduction="sum")(inference_outputs["prob"], perturbations)
            # * lamda
            * final_adv_penalty
        )

        weighted_kl_local = (
            kl_divergence_z_n * kl_weight  # + h_kl_weight * kl_divergence_v_n
        )
        weighted_kl_global = g_kl_weight * kl_divergence_u_q.sum()

        # make programs non-overlapping
        cross_corr_penalty = F.relu(weighted_cross_correlation(w_qk=self.w_qk, weights_q=self.b_q.sigmoid()))

        loss = torch.mean(rec_loss + weighted_kl_local) + weighted_kl_global + adv_loss + 100. * cross_corr_penalty

        kl_local = dict(
            kl_divergence_z=kl_divergence_z_n.mean(),
            # kl_divergence_h=kl_divergence_v_n.mean(),
        )

        # extra metrics for logging
        extra_metrics = {
            "adv_loss": adv_loss,
            # "kl_divergence_v_mean": kl_divergence_v_n.mean(),
            "kl_divergence_u_mean": kl_divergence_u_q.mean(),
            "adv_accuracy": adv_accuracy,
            "adv_penalty": final_adv_penalty,
            "cross_corr_penalty": cross_corr_penalty,
            "monitor": loss + self.b_q.sigmoid().median() * 1000.,
        }
        b_q_dict = logging_dict_from_tensor(self.b_q.sigmoid(), "b_q")
        # c_pq_dict = logging_dict_from_tensor(self.c_pq.sigmoid(), "c_pq")
        extra_metrics.update(b_q_dict)
        # extra_metrics.update(c_pq_dict)

        return LossOutput(
            loss=loss,
            reconstruction_loss=rec_loss,
            kl_local=kl_local,
            extra_metrics=extra_metrics,
        )


def weighted_cross_correlation(w_qk: torch.Tensor, weights_q: torch.Tensor) -> torch.Tensor:
    """Compute a weighted sum of cross-correlation between program definitions"""

    weights_q = F.normalize(weights_q, p=1, dim=0)
    e_qk = F.normalize(w_qk, p=2, dim=-1)
    z_qk = e_qk * weights_q.unsqueeze(-1)
    abs_corr_qq = torch.matmul(z_qk, z_qk.t()).abs()
    return abs_corr_qq.sum() - torch.trace(abs_corr_qq)


def logging_dict_from_tensor(x: torch.Tensor, name: str) -> Dict[str, torch.Tensor]:
    """Take a > zero-dimensional tensor and flatten, returning a dictionary
    with each value being a 0-d tensor, appropriate for logging."""
    flat_x = x.flatten()
    return {f"{name}_{i}": flat_x[i] for i in range(len(flat_x))}


# https://pytorch.org/docs/stable/distributions.html#torch.distributions.kl.register_kl
@register_kl(Delta, Distribution)
def kl_delta_any(delta_dist, p):
    return -1 * p.log_prob(delta_dist.mean)
