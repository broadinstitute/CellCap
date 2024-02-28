"""The CellCap scvi-tools module"""

import logging
import numpy as np
import pandas as pd
from anndata import AnnData

import torch
import torch.nn.functional as F


from scvi import REGISTRY_KEYS
from scvi.train import TrainRunner
from scvi.train._callbacks import SaveBestState
from scvi.model.base import (
    UnsupervisedTrainingMixin,
    BaseModelClass,
    RNASeqMixin,
    VAEMixin,
)

from scvi.data import AnnDataManager
from scvi.utils import setup_anndata_dsp
from scvi.train._trainingplans import TrainingPlan
from scvi.data.fields import (
    LayerField,
    ObsmField,
)

from .training_plan import DataSplitter

from .model import CellCapModel
from typing import Optional, Union, Literal

torch.backends.cudnn.benchmark = True
logger = logging.getLogger(__name__)

class CellCap(RNASeqMixin, VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 20,
        n_layers: int = 4,
        dropout_rate: float = 0.25,
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
            "{}, dispersion: {}, latent_distribution: {}"
        ).format(
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
            dispersion,
            latent_distribution,
        )

        self.n_latent = n_latent
        self.use_batch_norm = use_batch_norm
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
        w = torch.matmul(F.softplus(self.module.H_pq), self.module.w_qk.tanh())
        loadings = torch.Tensor.cpu(w).detach().numpy()

        return loadings

    def get_resp_loadings(self) -> pd.DataFrame:
        w = self.module.w_qk.tanh()
        loadings = torch.Tensor.cpu(w).detach().numpy()

        return loadings

    def get_covar_loadings(self) -> pd.DataFrame:
        w = self.module.w_covar_dk
        loadings = torch.Tensor.cpu(w).detach().numpy()
        loadings = pd.DataFrame(loadings.T)

        return loadings

    def get_h(self) -> pd.DataFrame:
        w = F.softplus(self.module.H_pq)
        w = torch.Tensor.cpu(w).detach().numpy()

        return w

    def get_ard(self) -> pd.DataFrame:
        w = self.module.alpha_q.sigmoid()
        w = torch.Tensor.cpu(w).detach().numpy()

        return w

    def get_H_key(self) -> pd.DataFrame:
        w = self.module.H_key
        w = torch.Tensor.cpu(w).detach().numpy()

        return w

    @torch.no_grad()
    def get_h_attn(
        self,
        adata: Optional[AnnData] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Get the inferred H_attn for each cell, which is the usage of each
        response program after attention is taken into account
        """

        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")

        adata = self._validate_anndata(adata)
        post = self._make_data_loader(adata=adata, batch_size=batch_size, shuffle=False)
        h_attn = []
        for tensors in post:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            out = outputs["H_attn"]
            h_attn.append(out.detach().cpu())

        return np.array(torch.cat(h_attn))

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

        return np.array(torch.cat(embedding)), np.array(torch.cat(h)), np.array(torch.cat(attn))

    @torch.no_grad()
    def get_covar_embedding(
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
            delta_z_covar = outputs["delta_z_covar"]
            embedding += [delta_z_covar.cpu()]
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
        predictions = []
        for tensors in post:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            generative_inputs = self.module._get_generative_input(tensors, outputs)
            outputs = self.module.generative(**generative_inputs)
            out = outputs["px"].mean  # .sample()
            predictions += [out.cpu()]
        return np.array(torch.cat(predictions))

    @torch.no_grad()
    def predict_basal(
            self,
            adata: Optional[AnnData] = None,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:
        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")

        adata = self._validate_anndata(adata)
        post = self._make_data_loader(adata=adata, batch_size=batch_size)
        predictions = []
        for tensors in post:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            outputs['delta_z'] = torch.zeros_like(outputs['delta_z'])
            outputs['delta_z_covar'] = torch.zeros_like(outputs['delta_z_covar'])
            generative_inputs = self.module._get_generative_input(tensors, outputs)
            outputs = self.module.generative(**generative_inputs)
            out = outputs["px"].mean  # .sample()
            predictions += [out.cpu()]
        return np.array(torch.cat(predictions))

    @torch.no_grad()
    def predict_pert(
            self,
            adata: Optional[AnnData] = None,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:
        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")

        adata = self._validate_anndata(adata)
        post = self._make_data_loader(adata=adata, batch_size=batch_size)
        predictions = []
        all_predictions = []
        for tensors in post:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            generative_inputs = self.module._get_generative_input(tensors, outputs)
            gen_outputs = self.module.generative(**generative_inputs)
            out = gen_outputs["px"].mean  # .sample() #
            all_predictions += [out.cpu()]

            outputs['delta_z'] = torch.zeros_like(outputs['delta_z'])
            outputs['delta_z_covar'] = torch.zeros_like(outputs['delta_z_covar'])
            generative_inputs = self.module._get_generative_input(tensors, outputs)
            gen_outputs = self.module.generative(**generative_inputs)
            out = gen_outputs["px"].mean  # .sample() #
            predictions += [out.cpu()]
        return np.array(torch.cat(all_predictions)) - np.array(torch.cat(predictions))

    def train(
        self,
        max_epochs: int = 500,
        lr: float = 1e-3,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        weight_decay: float = 1e-5,
        eps: float = 1e-08,
        early_stopping: bool = True,
        save_best: bool = True,
        check_val_every_n_epoch: Optional[int] = None,
        n_steps_kl_warmup: Optional[int] = None,
        n_epochs_kl_warmup: Optional[int] = 50,
        plan_kwargs: Optional[dict] = None,
        weighted_sampler: Optional[bool] = False,
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
            weighted_sampler=weighted_sampler,
        )
         
        training_plan = TrainingPlan(self.module, reduce_lr_on_plateau=True, **plan_kwargs)
        
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

    def setup_training(
        self,
        lamda: float = 1.0,
        kl_weight: float = 1.0,
        rec_weight: float = 2.0,
        ard_kl_weight: float = 0.2,
    ):
        self.module.lamda = lamda
        self.module.kl_weight = kl_weight
        self.module.rec_weight = rec_weight
        self.module.ard_kl_weight = ard_kl_weight

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: str,
        target_key: str,
        covar_key: str,
        **kwargs,
    ):
        """
        %(summary)s.

        Parameters
        ----------
        %(param_layer)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            ObsmField(registry_key="TARGET_KEY", attr_key=target_key),
            ObsmField(registry_key="COVAR_KEY", attr_key=covar_key),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args,
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
