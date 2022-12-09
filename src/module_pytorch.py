"""Module for scvi-tools written in pytorch"""

import torch
import torch.distributions as dist

from scvi import REGISTRY_KEYS
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data

from .nn import Encoder, LinearDecoder, Classifier, GradientReversal


class TorchLatentSpaceAttention(BaseModuleClass):
    """
    Skeleton Variational auto-encoder model.

    Here we implement a basic version of scVI's underlying VAE [Lopez18]_.
    This implementation is for instructional purposes only.

    Parameters
    ----------
    n_input
        Number of input genes
    n_latent
        Dimensionality of the latent space
    """

    def __init__(
        self,
        n_input: int,
        n_latent: int = 10,
    ):
        # super().__init__()
        # # in the init, we create the parameters of our elementary stochastic computation unit.
        #
        # # First, we setup the parameters of the generative model
        # self.decoder = LinearDecoderSCVI(n_input=n_latent, n_output=n_input, "softmax")
        # self.log_theta = torch.nn.Parameter(torch.randn(n_input))
        #
        # # Second, we setup the parameters of the variational distribution
        # self.mean_encoder = MyNeuralNet(n_input, n_latent, "none")
        # self.var_encoder = MyNeuralNet(n_input, n_latent, "exp")
        pass

    def _get_inference_input(self, tensors):
        """Parse the dictionary to get appropriate args"""
        # # let us fetch the raw counts, and add them to the dictionary
        # x = tensors[REGISTRY_KEYS.X_KEY]
        #
        # input_dict = dict(x=x)
        # return input_dict
        pass

    @auto_move_data
    def inference(self, x):
        """
        High level inference method.

        Runs the inference (encoder) model.
        """
        # # log the input to the variational distribution for numerical stability
        # x_ = torch.log(1 + x)
        # # get variational parameters via the encoder networks
        # qz_m = self.mean_encoder(x_)
        # qz_v = self.var_encoder(x_)
        # # get one sample to feed to the generative model
        # # under the hood here is the Reparametrization trick (Rsample)
        # z = dist.Normal(qz_m, torch.sqrt(qz_v)).rsample()
        #
        # outputs = dict(qz_m=qz_m, qz_v=qz_v, z=z)
        # return outputs
        pass

    def _get_generative_input(self, tensors, inference_outputs):
        # z = inference_outputs["z"]
        # x = tensors[REGISTRY_KEYS.X_KEY]
        # # here we extract the number of UMIs per cell as a known quantity
        # library = torch.sum(x, dim=1, keepdim=True)
        #
        # input_dict = {
        #     "z": z,
        #     "library": library,
        # }
        # return input_dict
        pass

    @auto_move_data
    def generative(self, z, library):
        """Runs the generative model."""

        # # get the "normalized" mean of the negative binomial
        # px_scale = self.decoder(z)
        # # get the mean of the negative binomial
        # px_rate = library * px_scale
        # # get the dispersion parameter
        # theta = torch.exp(self.log_theta)
        #
        # return dict(
        #     px_scale=px_scale, theta=theta, px_rate=px_rate
        # )
        pass

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
    ):

        # # here, we would like to form the ELBO. There are two terms:
        # #   1. one that pertains to the likelihood of the data
        # #   2. one that pertains to the variational distribution
        # # so we extract all the required information
        # x = tensors[REGISTRY_KEYS.X_KEY]
        # px_rate = generative_outputs["px_rate"]
        # theta = generative_outputs["theta"]
        # qz_m = inference_outputs["qz_m"]
        # qz_v = inference_outputs["qz_v"]
        #
        # # term 1
        # # the pytorch NB distribution uses a different parameterization
        # # so we must apply a quick transformation (included in scvi-tools, but here we use the pytorch code)
        # nb_logits = (px_rate + 1e-4).log() - (theta + 1e-4).log()
        # log_lik = dist.NegativeBinomial(total_count=theta, logits=nb_logits).log_prob(x).sum(dim=-1)
        #
        # # term 2
        # prior_dist = dist.Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v))
        # var_post_dist = dist.Normal(qz_m, torch.sqrt(qz_v))
        # kl_divergence = dist.kl_divergence(var_post_dist, prior_dist).sum(dim=1)
        #
        # elbo = log_lik - kl_divergence
        # loss = torch.mean(-elbo)
        # return LossRecorder(loss, -log_lik, kl_divergence, 0.0)
        pass
