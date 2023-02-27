import logging
import numpy as np
import pandas as pd
from anndata import AnnData

import torch
from torch import logsumexp
import torch.nn.functional as F
from torch.distributions import Normal, Poisson
from torch.distributions import kl_divergence as kl

from scvi import REGISTRY_KEYS
from scvi._compat import Literal
from scvi.train import TrainRunner
from scvi.dataloaders import DataSplitter
from scvi.train._callbacks import SaveBestState
from scvi.nn import Encoder, LinearDecoderSCVI, one_hot
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.distributions import ZeroInflatedNegativeBinomial, NegativeBinomial
from scvi.model.base import UnsupervisedTrainingMixin, BaseModelClass, RNASeqMixin, VAEMixin

from scvi.data import AnnDataManager
from scvi.utils import setup_anndata_dsp
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
    ObsmField,
)

from typing import Callable, Iterable, Optional, List, Union, Tuple, Dict

from .nn.drugencoder import DrugEncoder
from .nn.donorencoder import DonorEncoder
from .nn.attention import DotProductAttention
from .nn.advclassifier import AdvNet
from .nn.autoreldetermin import ARDregularizer

from .utils import entropy
from .training_plan import FactorTrainingPlanA

torch.backends.cudnn.benchmark = True
logger = logging.getLogger(__name__)

##VAE base
class LINEARVAE(BaseModuleClass):

    def __init__(
            self,
            n_input: int,
            n_labels: int = 0,
            n_hidden: int = 128,
            n_latent: int = 10,
            n_layers: int = 1,
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
            use_size_factor_key: bool = False,
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
        self.n_labels = n_labels
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.n_drug = n_drug
        self.n_control = n_control
        self.n_prog = n_prog
        self.n_donor = n_donor
        self.n_target = n_target
        self.use_size_factor_key = use_size_factor_key
        self.use_observed_lib_size = True  # use_size_factor_key or use_observed_lib_size
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

        self.d_encoder = DrugEncoder(n_latent, n_drug, n_prog ,key=False)
        self.c_encoder = DrugEncoder(n_latent, n_control, n_prog ,key=False)
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
        d = tensors['COND_KEY']
        c = tensors['CONT_KEY']
        donor = tensors['DONOR_KEY']

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
    def inference(self, x, d, c, donor,
                  # batch_index, cat_covs=None,
                  n_samples=1):
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
            ql_m = ql_m.unsqueeze(0).expand((n_samples, ql_m.size(0), ql_m.size(1)))
            ql_v = ql_v.unsqueeze(0).expand((n_samples, ql_v.size(0), ql_v.size(1)))
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

        outputs = dict(z=z, qz_m=qz_m, qz_v=qz_v, ql_m=ql_m, ql_v=ql_v,
                       prob=prob, Zp=Zp, Zc=Zc, Zd=Zd, library=library,
                       attP=attP, attC=attC,
                       alpha_ip_d=alpha_ip_d, alpha_ip_c=alpha_ip_c,
                       )
        return outputs

    @auto_move_data
    def generative(self, z, Zp, Zc, Zd, library,
                   # batch_index,
                   # cont_covs=None,
                   # cat_covs=None,
                   # size_factor=None,
                   y=None, transform_batch=None):
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

        advers_loss = torch.nn.BCELoss(reduction='sum')(inference_outputs["prob"], l)
        ent_penalty = entropy(generative_outputs["zA"])

        loss = torch.mean(
            reconst_loss * 0.5 + weighted_kl_local + advers_loss + ent_penalty * 0.2
        )

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

##CellCap
class CellCap(RNASeqMixin, VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):

    def __init__(
            self,
            adata: AnnData,
            n_hidden: int = 128,
            n_latent: int = 10,
            n_layers: int = 1,
            dropout_rate: float = 0.1,
            # n_batch: int = 0,
            dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
            gene_likelihood: Literal["zinb", "nb", "poisson"] = "nb",
            latent_distribution: Literal["normal", "ln"] = "normal",
            log_variational: bool = True,
            use_batch_norm: bool = True,
            bias: bool = False,
            **model_kwargs,
    ):
        super(CellCap, self).__init__(adata)
        self.module = LINEARVAE(
            n_input=self.summary_stats["n_vars"],
            # n_batch=self.summary_stats["n_batch"],
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

    def get_pert_loadings(self) -> pd.DataFrame:  # TO DO

        weights = []
        for p in self.module.d_encoder.drug_weights.parameters():
            weights.append(p)
        w = weights[0]
        w = F.normalize(w, p=2, dim=2)
        loadings = torch.Tensor.cpu(w).detach().numpy()

        return loadings

    def get_ard_loadings(self) -> pd.DataFrame:  # TO DO

        weights = []
        for p in self.module.ard_d.ard_dist.parameters():
            weights.append(p)
        w = weights[0]
        loadings = torch.Tensor.cpu(w).detach().numpy()

        return loadings

    def get_donor_loadings(self) -> pd.DataFrame:  # TO DO

        weights = []
        for p in self.module.donor_encoder.drug_weights.parameters():
            weights.append(p)
        w = weights[0]
        w = F.normalize(w, p=2, dim=1)
        loadings = torch.Tensor.cpu(w).detach().numpy()
        loadings = pd.DataFrame(loadings.T)

        return loadings

    @torch.no_grad()
    def get_latent_embedding(
            self,
            adata: Optional[AnnData] = None,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:

        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")

        adata = self._validate_anndata(adata)
        post = self._make_data_loader(
            adata=adata, batch_size=batch_size
        )  # indices=indices,
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

        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")

        adata = self._validate_anndata(adata)
        post = self._make_data_loader(
            adata=adata, batch_size=batch_size
        )  # indices=indices,
        embedding = []
        atts = []
        for tensors in post:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            Zp = outputs["Zp"]
            embedding += [Zp.cpu()]

            attP = outputs["attP"]
            atts += [attP.cpu()]

        return np.array(torch.cat(embedding)), np.array(torch.cat(atts))

    @torch.no_grad()
    def get_control_embedding(
            self,
            adata: Optional[AnnData] = None,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:

        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")

        adata = self._validate_anndata(adata)
        post = self._make_data_loader(
            adata=adata, batch_size=batch_size
        )
        embedding = []
        atts = []
        for tensors in post:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            Zp = outputs["Zc"]
            embedding += [Zp.cpu()]

            attC = outputs["attC"]
            atts += [attC.cpu()]

        return np.array(torch.cat(embedding)), np.array(torch.cat(atts))

    @torch.no_grad()
    def get_donor_embedding(
            self,
            adata: Optional[AnnData] = None,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:

        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")

        adata = self._validate_anndata(adata)
        post = self._make_data_loader(
            adata=adata, batch_size=batch_size
        )  # indices=indices,
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

        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")

        adata = self._validate_anndata(adata)
        post = self._make_data_loader(
            adata=adata, batch_size=batch_size
        )  # indices=indices,
        embedding = []
        for tensors in post:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            z = outputs["z"]
            Zp = outputs["Zp"]
            embedding += [z.cpu() + Zp.cpu()]
        return np.array(torch.cat(embedding))

    @torch.no_grad()
    def predict(
            self,
            adata: Optional[AnnData] = None,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:

        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")

        adata = self._validate_anndata(adata)
        post = self._make_data_loader(
            adata=adata, batch_size=batch_size
        )
        preditions = []
        for tensors in post:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            generative_inputs = self.module._get_generative_input(tensors, outputs)
            outputs = self.module.generative(**generative_inputs)
            out = outputs['px'].sample()
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
        update_dict = dict(
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            n_steps_kl_warmup=n_steps_kl_warmup,
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

        training_plan = FactorTrainingPlanA(self.module, discriminator=True, scale_tc_loss=1.0, **plan_kwargs)

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

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
            cls,
            adata: AnnData,
            layer: str,
            pert_key: str,
            cond_key: str,
            cont_key: str,
            target_key: str,
            donor_key: str,
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
            CategoricalObsField(registry_key="PERT_KEY", obs_key=pert_key),
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
