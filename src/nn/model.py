import logging
import numpy as np
import pandas as pd
from anndata import AnnData

import torch
import torch.nn.functional as F

from scvi import REGISTRY_KEYS
from scvi._compat import Literal
from scvi.train import TrainRunner
from scvi.dataloaders import DataSplitter
from scvi.train._callbacks import SaveBestState
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

from typing import Optional, List, Union

torch.backends.cudnn.benchmark = True

from .base import LINEARVAE
from .advclassifier import FactorTrainingPlan

class CellCap(RNASeqMixin, VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):

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
        super(CellCap, self).__init__(adata)
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
        )  # indices=indices,
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
        return np.array(torch.cat(embedding))  # (F.normalize(, p=2, dim=1))

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
        return np.array(torch.cat(embedding))  # (F.normalize(, p=2, dim=1))

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

        training_plan = FactorTrainingPlan(self.module, discriminator=True, scale_tc_loss=1.0, **plan_kwargs)

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
