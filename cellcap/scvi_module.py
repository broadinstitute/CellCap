"""The CellCap scvi-tools module"""

import logging
import numpy as np
import pandas as pd
from anndata import AnnData

import torch

from scvi import REGISTRY_KEYS
from scvi.train import TrainRunner
from scvi.dataloaders import DataSplitter
from scvi.train._callbacks import SaveBestState
from scvi.model.base import (
    UnsupervisedTrainingMixin,
    BaseModelClass,
    RNASeqMixin,
    VAEMixin,
)

from scvi.data import AnnDataManager
from scvi.utils import setup_anndata_dsp
from scvi.data.fields import (
    CategoricalObsField,
    LayerField,
    ObsmField,
)

from typing import Optional, Union, Literal

from scvi.train._trainingplans import TrainingPlan

from .model import CellCapModel

torch.backends.cudnn.benchmark = True
logger = logging.getLogger(__name__)


class CellCap(RNASeqMixin, VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-label", "gene-cell"] = "gene",
        latent_distribution: Literal["normal", "ln"] = "normal",
        use_batch_norm: bool = True,
        **model_kwargs,
    ):
        super(CellCap, self).__init__(adata)
        self.module = CellCapModel(
            n_input=self.summary_stats["n_vars"],
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers_encoder=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            latent_distribution=latent_distribution,
            **model_kwargs,
        )
        self._model_summary_string = (
            "CellCap Model with the following params: \n"
            "n_hidden: {}, n_latent: {}, n_layers: {}, dropout_rate: "
            "{}, dispersion: {}, latent_distribution: {}"  # gene_likelihood: {},
        ).format(
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
            dispersion,
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

        return loadings

    def get_pert_loadings(self) -> pd.DataFrame:
        w = torch.matmul(self.module.H_pq.sigmoid(), self.module.w_qk)
        loadings = torch.Tensor.cpu(w).detach().numpy()

        return loadings

    def get_resp_loadings(self) -> pd.DataFrame:
        w = self.module.w_qk
        loadings = torch.Tensor.cpu(w).detach().numpy()

        return loadings

    def get_donor_loadings(self) -> pd.DataFrame:
        w = self.module.w_donor_dk
        loadings = torch.Tensor.cpu(w).detach().numpy()
        loadings = pd.DataFrame(loadings.T)

        return loadings

    def get_h(self) -> pd.DataFrame:
        w = self.module.H_pq.sigmoid()
        w = torch.Tensor.cpu(w).detach().numpy()

        return w

    def get_ard(self) -> pd.DataFrame:
        w = self.module.log_alpha_pq.sigmoid()
        w = torch.Tensor.cpu(w).detach().numpy()

        return w

    @torch.no_grad()
    def get_latent_embedding(
        self,
        adata: Optional[AnnData] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")

        adata = self._validate_anndata(adata)
        post = self._make_data_loader(adata=adata, batch_size=batch_size)
        embedding = []
        for tensors in post:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            z = outputs["z_basal"]
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
        post = self._make_data_loader(adata=adata, batch_size=batch_size)
        embedding = []
        h = []
        attn = []
        for tensors in post:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            delta_z = outputs["delta_z"]
            embedding += [delta_z.cpu()]

            H_m = outputs["H_attn"]
            h += [H_m.cpu()]

            a = outputs["attn"]
            attn += [a.cpu()]

        return (
            np.array(torch.cat(embedding)),
            np.array(torch.cat(h)),
            np.array(torch.cat(attn)),
        )

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
            delta_z_donor = outputs["delta_z_donor"]
            embedding += [delta_z_donor.cpu()]
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
        post = self._make_data_loader(adata=adata, batch_size=batch_size)
        embedding = []
        for tensors in post:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            z_basal = outputs["z_basal"]
            delta_z = outputs["delta_z"]
            embedding += [z_basal.cpu() + delta_z.cpu()]
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
        post = self._make_data_loader(adata=adata, batch_size=batch_size)
        preditions = []
        for tensors in post:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            generative_inputs = self.module._get_generative_input(tensors, outputs)
            outputs = self.module.generative(**generative_inputs)
            out = outputs["px"].sample()
            preditions += [out.cpu()]
        return np.array(torch.cat(preditions))

    def train(
        self,
        max_epochs: int = 500,
        lr: float = 1e-3,
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

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: str,
        target_key: str,
        donor_key: str,
        **kwargs,
    ):
        """
        %(summary)s.

        Parameters
        ----------
        %(param_layer)s
        target_key: Key for adata.obsm containing a one-hot encoding of
            perturbation information
        donor_key: Key for adata.obsm containing a one-hot encoding of donor
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            ObsmField(registry_key="TARGET_KEY", attr_key=target_key),
            ObsmField(registry_key="DONOR_KEY", attr_key=donor_key),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
