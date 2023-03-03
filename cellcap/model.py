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

from .nn.drugencoder import DrugEncoder
from .nn.donorencoder import DonorEncoder
from .nn.attention import DotProductAttention
from .nn.advclassifier import AdvNet
from .nn.autoreldetermin import ARDregularizer
from .mixins import CellCapMixin

from .utils import entropy

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
        n_target: int = 5,
        n_control: int = 3,
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
        self.n_control = n_control
        self.n_prog = n_prog
        self.n_donor = n_donor
        self.n_target = n_target

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

        # l encoder goes from n_input-dimensional data to 1-d library size
        # TODO: this l_encoder is only here so refactor tests can run
        # TODO: it is not used and can be deleted
        self.l_encoder = Encoder(
            n_input,
            1,
            n_layers=1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm in ["encoder", "both"],
            use_layer_norm=False,
        )

        self.d_encoder = DrugEncoder(n_latent, n_drug, n_prog, key=False)
        self.c_encoder = DrugEncoder(n_latent, n_control, n_prog, key=False)
        self.d_encoder_key = DrugEncoder(n_latent, n_drug, n_prog, key=True)
        self.c_encoder_key = DrugEncoder(n_latent, n_control, n_prog, key=True)

        self.donor_encoder = DonorEncoder(n_latent, n_donor)

        self.ard_d = ARDregularizer(n_drug, n_prog)
        self.ard_c = ARDregularizer(n_control, n_prog)

        self.attention = DotProductAttention()

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
            out_dim=self.n_target,
        )

    def _get_inference_input(self, tensors):
        x = tensors[REGISTRY_KEYS.X_KEY]
        d = tensors["COND_KEY"]
        c = tensors["CONT_KEY"]
        donor = tensors["DONOR_KEY"]

        input_dict = dict(
            x=x,
            d=d,
            c=c,
            donor=donor,
        )
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        Zp = inference_outputs["Zp"]
        Zc = inference_outputs["Zc"]
        Zd = inference_outputs["Zd"]
        library = inference_outputs["library"]

        input_dict = dict(
            z=z,
            Zp=Zp,
            Zc=Zc,
            Zd=Zd,
            library=library,
        )
        return input_dict

    @auto_move_data
    def inference(self, x, d, c, donor, n_samples=1):
        """
        High level inference method.

        Runs the inference (encoder) model.
        """
        x_ = x
        library = torch.log(x.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)
        encoder_input = x_

        qz_m, qz_v, z = self.z_encoder(encoder_input)

        # TODO: after refactor, we can delete these lines, as l_encoder is not used
        ql_m, ql_v, library_encoded = self.l_encoder(
            encoder_input,
        )

        Zd = self.donor_encoder(donor)

        Zp = self.d_encoder(d)
        Zp_key = self.d_encoder_key(d)
        Zc = self.c_encoder(c)
        Zc_key = self.c_encoder_key(c)

        alpha_ip_d = self.ard_d(d)
        alpha_ip_c = self.ard_c(c)

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.z_encoder.z_transformation(untran_z)
            library = library.unsqueeze(0).expand(
                (n_samples, library.size(0), library.size(1))
            )

        Zp, attP = self.attention(z.unsqueeze(1), Zp_key, Zp)
        Zp = Zp.squeeze(1)
        attP = attP.squeeze(1)
        Zc, attC = self.attention(z.unsqueeze(1), Zc_key, Zc)
        Zc = Zc.squeeze(1)
        attC = attC.squeeze(1)

        prob = self.classifier(z)

        z = F.normalize(z, p=2, dim=1)
        Zp = F.normalize(Zp, p=2, dim=1)
        Zc = F.normalize(Zc, p=2, dim=1)
        Zd = F.normalize(Zd, p=2, dim=1)

        outputs = dict(
            z=z,
            qz_m=qz_m,
            qz_v=qz_v,
            prob=prob,
            Zp=Zp,
            Zc=Zc,
            Zd=Zd,
            library=library,
            attP=attP,
            attC=attC,
            alpha_ip_d=alpha_ip_d,
            alpha_ip_c=alpha_ip_c,
        )
        return outputs

    @auto_move_data
    def generative(self, z, Zp, Zc, Zd, library, y=None, transform_batch=None):
        """Runs the generative model."""
        # Likelihood distribution
        zA = z + Zp + Zc + Zd
        decoder_input = zA

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
        pz = Normal(torch.zeros_like(zA), torch.ones_like(zA))
        return dict(zA=zA, px=px, pl=pl, pz=pz)

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ):
        x = tensors[REGISTRY_KEYS.X_KEY]
        label = tensors["TARGET_KEY"]

        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]

        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)

        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(
            dim=1
        )

        kl_divergence_l = 0.0

        reconst_loss = -generative_outputs["px"].log_prob(x).sum(-1)

        kl_local_for_warmup = kl_divergence_z
        kl_local_no_warmup = kl_divergence_l
        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

        advers_loss = torch.nn.BCELoss(reduction="sum")(
            inference_outputs["prob"], label
        )
        ent_penalty = entropy(generative_outputs["zA"])

        loss = torch.mean(
            reconst_loss * 0.5 + weighted_kl_local + advers_loss + ent_penalty * 0.2
        )

        kl_local = dict(
            kl_divergence_l=kl_divergence_l, kl_divergence_z=kl_divergence_z
        )
        kl_global = torch.tensor(0.0)

        return LossRecorder(loss, reconst_loss, kl_local, kl_global)
