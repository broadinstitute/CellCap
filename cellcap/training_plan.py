"""Training plan with arbitrary logging"""
from torch.optim.lr_scheduler import ReduceLROnPlateau

from scvi._compat import Literal
from scvi.train._trainingplans import TrainingPlan
from scvi.module.base._base_module import BaseModuleClass

from typing import Optional, Union

from .utils import _METRICS_TO_LOG
from .nn.advclassifier import AdvNet

def _compute_kl_weight(
        epoch: int,
        step: int,
        n_epochs_kl_warmup: Optional[int],
        n_steps_kl_warmup: Optional[int],
        min_weight: Optional[float] = None,
) -> float:
    epoch_criterion = n_epochs_kl_warmup is not None
    step_criterion = n_steps_kl_warmup is not None
    if epoch_criterion:
        kl_weight = min(1.0, epoch / n_epochs_kl_warmup)
    elif step_criterion:
        kl_weight = min(1.0, step / n_steps_kl_warmup)
    else:
        kl_weight = 1.0
    if min_weight is not None:
        kl_weight = max(kl_weight, min_weight)
    return kl_weight


class FactorTrainingPlanA(TrainingPlan):

    def __init__(
            self,
            module: BaseModuleClass,
            lr=1e-3,
            weight_decay=1e-6,
            n_steps_kl_warmup: Union[int, None] = None,
            n_epochs_kl_warmup: Union[int, None] = 400,
            reduce_lr_on_plateau: bool = False,
            lr_factor: float = 0.6,
            lr_patience: int = 30,
            lr_threshold: float = 0.0,
            lr_scheduler_metric: Literal[
                "elbo_validation", "reconstruction_loss_validation", "kl_local_validation"
            ] = "elbo_validation",
            lr_min: float = 0,
            discriminator: Union[bool, AdvNet] = False,
            scale_tc_loss: Union[float, Literal["auto"]] = "auto",
            **loss_kwargs,
    ):
        super().__init__(
            module=module,
            lr=lr,
            weight_decay=weight_decay,
            n_steps_kl_warmup=n_steps_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            lr_threshold=lr_threshold,
            lr_scheduler_metric=lr_scheduler_metric,
            lr_min=lr_min,
        )
        if discriminator is True:
            self.n_output_classifier = self.module.n_batch
            self.discriminator = AdvNet(
                in_feature=self.module.n_latent,
                hidden_size=128,
                out_dim=2
            )
        else:
            self.discriminator = discriminator
        self.scale_tc_loss = scale_tc_loss

    def tc_loss(self, z):
        Dz = self.discriminator(z)
        tc_loss = (Dz[:, :1] - Dz[:, 1:]).mean()
        return tc_loss

    def discriminator_loss(self, z):

        Dzb = self.discriminator.forward(z, if_activation=False, reverse=False)
        zperm = permute_dims(z)
        Dperm = self.discriminator.forward(zperm.detach(), if_activation=False, reverse=False)

        ones = torch.ones(z.shape[0], dtype=torch.long, device=z.device)
        zeros = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
        d_loss1 = torch.nn.CrossEntropyLoss(reduction='mean')(Dzb, zeros)
        d_loss2 = torch.nn.CrossEntropyLoss(reduction='mean')(Dperm, ones)
        loss = (d_loss1 + d_loss2) * 0.5

        return loss

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        kappa = (
            1 - self.kl_weight
            if self.scale_tc_loss == "auto"
            else self.scale_tc_loss
        )
        if optimizer_idx == 0:

            loss_kwargs = dict(kl_weight=self.kl_weight)
            inference_outputs, _, scvi_loss = self.forward(
                batch, loss_kwargs=loss_kwargs
            )
            loss = scvi_loss.loss
            # fool classifier if doing adversarial training
            if kappa > 0 and self.discriminator is not False:
                z = inference_outputs["z"]
                tc_loss = self.tc_loss(z)
                loss += tc_loss * kappa

            self.log("train_loss", loss, on_epoch=True)
            self.compute_and_log_metrics(scvi_loss, self.elbo_train)
            return loss

        if optimizer_idx == 1:
            inference_inputs = self.module._get_inference_input(batch)
            outputs = self.module.inference(**inference_inputs)
            z = outputs["z"]
            loss = self.discriminator_loss(z)
            loss *= kappa

            return loss

        if optimizer_idx == 2:
            inference_inputs = self.module._get_inference_input(batch)
            outputs = self.module.inference(**inference_inputs)

            ard_reg_d = Normal(loc=0., scale=1. / outputs["alpha_ip_d"]).log_prob(outputs["attP"]).mean()
            ard_reg_c = Normal(loc=0., scale=1. / outputs["alpha_ip_c"]).log_prob(outputs["attC"]).mean()
            loss = (ard_reg_d + ard_reg_c) * 0.00001

            return loss

    def configure_optimizers(self):
        params1 = filter(lambda p: p.requires_grad, self.module.parameters())
        optimizer1 = torch.optim.AdamW(
            params1, lr=self.lr, eps=0.01, weight_decay=self.weight_decay
        )
        config1 = {"optimizer": optimizer1}
        if self.reduce_lr_on_plateau:
            scheduler1 = ReduceLROnPlateau(
                optimizer1,
                patience=self.lr_patience,
                factor=self.lr_factor,
                threshold=self.lr_threshold,
                min_lr=self.lr_min,
                threshold_mode="abs",
                verbose=True,
            )
            config1.update(
                {
                    "lr_scheduler": scheduler1,
                    "monitor": self.lr_scheduler_metric,
                },
            )

        params2 = filter(
            lambda p: p.requires_grad, self.discriminator.parameters()
        )
        optimizer2 = torch.optim.AdamW(
            params2, lr=self.lr, eps=0.01, weight_decay=self.weight_decay
        )  # or self.lr
        config2 = {"optimizer": optimizer2}

        Pa = list(self.module.ard_d.parameters()) + list(self.module.ard_c.parameters())
        Pb = list(self.module.d_encoder_key.parameters()) + list(self.module.c_encoder_key.parameters())
        params3 = filter(lambda p: p.requires_grad, Pa + Pb)
        optimizer3 = torch.optim.AdamW(
            params3, lr=self.lr, eps=0.01, weight_decay=self.weight_decay
        )
        config3 = {"optimizer": optimizer3}

        # bug in pytorch lightning requires this way to return
        opts = [config1.pop("optimizer"), config2["optimizer"], config3["optimizer"]]
        if "lr_scheduler" in config1:
            config1["scheduler"] = config1.pop("lr_scheduler")
            scheds = [config1]
            return opts, scheds
        else:
            return opts

        return config1


class FactorTrainingPlanB(TrainingPlan):

    def __init__(
            self,
            module: BaseModuleClass,
            lr=1e-3,
            weight_decay=1e-6,
            n_steps_kl_warmup: Union[int, None] = None,
            n_epochs_kl_warmup: Union[int, None] = 400,
            reduce_lr_on_plateau: bool = False,
            lr_factor: float = 0.6,
            lr_patience: int = 30,
            lr_threshold: float = 0.0,
            lr_scheduler_metric: Literal[
                "elbo_validation", "reconstruction_loss_validation", "kl_local_validation"
            ] = "elbo_validation",
            lr_min: float = 0,
            discriminator: Union[bool, AdvNet] = False,
            scale_tc_loss: Union[float, Literal["auto"]] = "auto",
            **loss_kwargs,
    ):
        super().__init__(
            module=module,
            lr=lr,
            weight_decay=weight_decay,
            n_steps_kl_warmup=n_steps_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            lr_threshold=lr_threshold,
            lr_scheduler_metric=lr_scheduler_metric,
            lr_min=lr_min,
        )
        if discriminator is True:
            self.n_output_classifier = self.module.n_batch
            self.discriminator = AdvNet(
                in_feature=self.module.n_latent,
                hidden_size=128,
                out_dim=2
            )
        else:
            self.discriminator = discriminator
        self.scale_tc_loss = scale_tc_loss

    def tc_loss(self, z):
        Dz = self.discriminator(z)
        tc_loss = (Dz[:, :1] - Dz[:, 1:]).mean()
        return tc_loss

    def discriminator_loss(self, z):

        Dzb = self.discriminator.forward(z, if_activation=False, reverse=False)
        zperm = permute_dims(z)
        Dperm = self.discriminator.forward(zperm.detach(), if_activation=False, reverse=False)

        ones = torch.ones(z.shape[0], dtype=torch.long, device=z.device)
        zeros = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
        d_loss1 = torch.nn.CrossEntropyLoss(reduction='mean')(Dzb, zeros)
        d_loss2 = torch.nn.CrossEntropyLoss(reduction='mean')(Dperm, ones)
        loss = (d_loss1 + d_loss2) * 0.5

        return loss

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        kappa = (
            1 - self.kl_weight
            if self.scale_tc_loss == "auto"
            else self.scale_tc_loss
        )
        if optimizer_idx == 0:
            loss_kwargs = dict(kl_weight=self.kl_weight)
            inference_outputs, _, scvi_loss = self.forward(
                batch, loss_kwargs=loss_kwargs
            )
            loss = scvi_loss.loss
            # fool classifier if doing adversarial training
            if kappa > 0 and self.discriminator is not False:
                z = inference_outputs["z"]
                tc_loss = self.tc_loss(z)
                loss += tc_loss * kappa

            self.log("train_loss", loss, on_epoch=True)
            self.compute_and_log_metrics(scvi_loss, self.elbo_train)
            return loss

        if optimizer_idx == 1:
            inference_inputs = self.module._get_inference_input(batch)
            outputs = self.module.inference(**inference_inputs)
            z = outputs["z"]
            loss = self.discriminator_loss(z)
            loss *= kappa

            return loss

    def configure_optimizers(self):
        params1 = filter(lambda p: p.requires_grad, self.module.parameters())
        optimizer1 = torch.optim.AdamW(
            params1, lr=self.lr, eps=0.01, weight_decay=self.weight_decay
        )
        config1 = {"optimizer": optimizer1}
        if self.reduce_lr_on_plateau:
            scheduler1 = ReduceLROnPlateau(
                optimizer1,
                patience=self.lr_patience,
                factor=self.lr_factor,
                threshold=self.lr_threshold,
                min_lr=self.lr_min,
                threshold_mode="abs",
                verbose=True,
            )
            config1.update(
                {
                    "lr_scheduler": scheduler1,
                    "monitor": self.lr_scheduler_metric,
                },
            )

        if self.discriminator is not False:
            params2 = filter(
                lambda p: p.requires_grad, self.discriminator.parameters()
            )
            optimizer2 = torch.optim.AdamW(
                params2, lr=self.lr, eps=0.01, weight_decay=self.weight_decay
            )  # or 1e-3
            config2 = {"optimizer": optimizer2}

            # bug in pytorch lightning requires this way to return
            opts = [config1.pop("optimizer"), config2["optimizer"]]
            if "lr_scheduler" in config1:
                config1["scheduler"] = config1.pop("lr_scheduler")
                scheds = [config1]
                return opts, scheds
            else:
                return opts

        return config1

# TODO: maybe not worth it to figure out this commented-out approach
# def training_epoch_end(cls, outputs):
#     cls.training_epoch_end(outputs=outputs)
#     for name, val in _METRICS_TO_LOG:
#         cls.log(name, val)
#
#
# def logging_wrapper(superclass):
#     """
#     Programmatically create subclasses which overwrite the
#     training_epoch_end() method.
#
#     Examples
#     --------
#
#     >>> training_plan = LoggedTrainingPlan(self.module, **plan_kwargs)
#     would be the same as
#     >>> training_plan = logging_wrapper(TrainingPlan)(self.module, **plan_kwargs)
#     except now you can also do
#     >>> training_plan = logging_wrapper(MyTrainingPlan)(self.module, **plan_kwargs)
#     """
#
#     _LoggedClass = type(f'Logged{superclass}', superclass, {})
#     _LoggedClass.training_epoch_end = classmethod(training_epoch_end)
#
#     return _LoggedClass


class LoggedTrainingPlan(TrainingPlan):

    def training_epoch_end(self, outputs):
        super().training_epoch_end(outputs=outputs)
        for name, val in _METRICS_TO_LOG:
            self.log(name, val)
