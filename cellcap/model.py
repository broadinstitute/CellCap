"""The CellCap model"""

import logging

import torch
import torch.nn.functional as F
from torch.distributions import Normal, Poisson
from torch.distributions import kl_divergence as kl

from scvi import REGISTRY_KEYS
from scvi._compat import Literal
from scvi.nn import Encoder, LinearDecoderSCVI, one_hot
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.distributions import ZeroInflatedNegativeBinomial, NegativeBinomial

from .mixins import CellCapMixin
from .nn.advclassifier import AdvNet

# from .utils import entropy

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
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        log_variational: bool = True,
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
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

        self.alpha_pq = torch.nn.Parameter(torch.ones(n_drug, n_prog))
        self.H_pq = torch.nn.Parameter(torch.rand(n_drug, n_prog))
        self.w_qk = torch.nn.Parameter(torch.rand(n_prog, n_latent))
        self.w_donor_dk = torch.nn.Parameter(torch.zeros(n_donor, n_latent))

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

        self.classifier = AdvNet(
            in_feature=n_latent,
            hidden_size=64,
            out_dim=self.n_drug,
        )

    def _get_inference_input(self, tensors):
        x = tensors[REGISTRY_KEYS.X_KEY]
        p = tensors["TARGET_KEY"]
        donor = tensors["DONOR_KEY"]

        input_dict = dict(
            x=x,
            p=p,
            donor=donor,
        )
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z_basal = inference_outputs["z_basal"]
        delta_z = inference_outputs["delta_z"]
        delta_z_donor = inference_outputs["delta_z_donor"]
        library = inference_outputs["library"]

        input_dict = dict(
            z_basal=z_basal,
            delta_z=delta_z,
            delta_z_donor=delta_z_donor,
            library=library,
        )
        return input_dict

    @auto_move_data
    def inference(self, x, p, donor, n_samples=1):
        """
        High level inference method.

        Runs the inference (encoder) model.
        """
        x_ = x
        library = torch.log(x.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)
        encoder_input = x_

        qz_m, qz_v, z_basal = self.z_encoder(encoder_input)

        h = torch.matmul(p, self.H_pq)
        delta_z = torch.matmul(h, self.w_qk)

        delta_z_donor = torch.matmul(donor, self.w_donor_dk)

        log_alpha_ip = torch.matmul(p, self.alpha_pq)
        alpha_ip = log_alpha_ip.exp()

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z_basal = self.z_encoder.z_transformation(untran_z)
            library = library.unsqueeze(0).expand(
                (n_samples, library.size(0), library.size(1))
            )

        prob = self.classifier(z_basal)

        z_basal = F.normalize(z_basal, p=2, dim=1)
        delta_z = F.normalize(delta_z, p=2, dim=1)
        delta_z_donor = F.normalize(delta_z_donor, p=2, dim=1)

        outputs = dict(
            z_basal=z_basal,
            qz_m=qz_m,
            qz_v=qz_v,
            prob=prob,
            delta_z=delta_z,
            delta_z_donor=delta_z_donor,
            library=library,
            h=h,
            alpha_ip=alpha_ip,
        )
        return outputs

    @auto_move_data
    def generative(self, z_basal, delta_z, delta_z_donor, library, y=None, transform_batch=None):
        """Runs the generative model."""
        # Likelihood distribution
        z = z_basal + delta_z + delta_z_donor
        decoder_input = z

        # for scvi
        categorical_input = tuple()
        size_factor = library

        px_scale, px_r, px_rate, px_dropout = self.decoder(
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
        pl = None
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))
        return dict(z=z, px=px, pl=pl, pz=pz)

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
        lamda: float = 2.0,  # coefficient of adversarial loss
    ):
        x = tensors[REGISTRY_KEYS.X_KEY]
        label = tensors["TARGET_KEY"]

        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]

        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)

        # reconstruction loss
        rec_loss = -generative_outputs["px"].log_prob(x).sum(-1)

        # KL divergence
        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(
            dim=1
        )

        kl_divergence_h = (
            Normal(loc=0.0, scale=1.0 / inference_outputs["alpha_ip"])
            .log_prob(inference_outputs["h"])
            .sum()
        )

        kl_divergence_delta = (
            Normal(loc=0.0, scale=0.1)
            .log_prob(inference_outputs["delta_z"])
            .sum()
        )

        kl_divergence_l = 0.0
        kl_local_for_warmup = kl_divergence_z
        kl_local_no_warmup = kl_divergence_l
        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

        # Adversarial loss
        adv_loss = torch.nn.BCELoss(reduction="sum")(
            inference_outputs["prob"], label
        )
        # ent_penalty = entropy(generative_outputs["z"])

        loss = torch.mean(
            rec_loss + weighted_kl_local + kl_divergence_h + kl_divergence_delta + lamda * adv_loss
        )

        kl_local = dict(
            kl_divergence_l=kl_divergence_l, kl_divergence_z=kl_divergence_z
        )
        kl_global = torch.tensor(0.0)

        return LossRecorder(loss, rec_loss, kl_local, kl_global)
