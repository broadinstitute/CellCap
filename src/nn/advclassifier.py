import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scvi.train._trainingplans import TrainingPlan
from scvi.module.base._base_module import BaseModuleClass
from scvi._compat import Literal
from .base import init_weights

from typing import Optional, Union


def permute_dims(z):
    assert z.dim() == 2

    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)


# Gradiant reverse
class GradientReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coeff, input):
        ctx.coeff = coeff
        return input

    @staticmethod
    def backward(ctx, grad_outputs):
        coeff = ctx.coeff
        return None, -coeff * grad_outputs


class GradientReverseModule(torch.nn.Module):
    def __init__(self, coeff_schedule):
        super(GradientReverseModule, self).__init__()
        self.coeff_schedule = coeff_schedule
        self.global_step = 0.0
        self.coeff = 0.0
        self.grl = GradientReverseLayer.apply

    def forward(self, x):
        self.coeff = self.coeff_schedule[int(self.global_step)]
        self.global_step += 1.0
        return self.grl(self.coeff, x)


class AdvNet(torch.nn.Module):
    def __init__(self, in_feature=20, hidden_size=20, out_dim=2):
        super(AdvNet, self).__init__()
        self.ad_layer1 = torch.nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = torch.nn.Linear(hidden_size, out_dim)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.norm1 = torch.nn.BatchNorm1d(hidden_size)
        self.norm2 = torch.nn.BatchNorm1d(hidden_size)
        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.25)
        self.sigmoid = torch.nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000
        self.grl = GradientReverseModule(
            torch.linspace(start=self.low, end=self.high, steps=self.max_iter),
        )

    def forward(self, x, reverse=True, if_activation=True):
        if reverse:
            x = self.grl(x)
        x = self.ad_layer1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        if if_activation:
            y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1


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


class FactorTrainingPlan(TrainingPlan):

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
