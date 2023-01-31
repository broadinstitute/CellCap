# """Module for scvi-tools written in pyro"""
#
# import torch
# import pyro
# import pyro.distributions as dist
#
# from scvi import REGISTRY_KEYS
# from scvi.module.base import PyroBaseModuleClass
#
# from .nn import Encoder, LinearDecoder, Classifier, GradientReversal
#
#
# class PyroLatentSpaceAttention(PyroBaseModuleClass):
#     """
#
#     Parameters
#     ----------
#     on_load_kwargs
#         Dictionary containing keyword args to use in ``self.on_load``.
#
#     """
#
#     def __init__(self, n_input, n_latent):
#         # super().__init__()
#         # self.n_latent = n_latent
#         # self.n_input = n_input
#         # # in the init, we create the parameters of our elementary stochastic computation unit.
#         #
#         # # First, we setup the parameters of the generative model
#         # self.decoder = MyNeuralNet(n_latent, n_input, "softmax")
#         # self.log_theta = torch.nn.Parameter(torch.randn(n_input))
#         #
#         # # Second, we setup the parameters of the variational distribution
#         # self.mean_encoder = MyNeuralNet(n_input, n_latent, "none")
#         # self.var_encoder = MyNeuralNet(n_input, n_latent, "exp")
#         pass
#
#     @staticmethod
#     def _get_fn_args_from_batch(tensor_dict):
#         # x = tensor_dict[REGISTRY_KEYS.X_KEY]
#         # library = torch.sum(x, dim=1, keepdim=True)
#         # return (x, library), {}
#         pass
#
#     def model(self, x, library):
#         # # register PyTorch module `decoder` with Pyro
#         # pyro.module("scvi", self)
#         # with pyro.plate("data", x.shape[0]):
#         #     # setup hyperparameters for prior p(z)
#         #     z_loc = x.new_zeros(torch.Size((x.shape[0], self.n_latent)))
#         #     z_scale = x.new_ones(torch.Size((x.shape[0], self.n_latent)))
#         #     # sample from prior (value will be sampled by guide when computing the ELBO)
#         #     z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
#         #     # get the "normalized" mean of the negative binomial
#         #     px_scale = self.decoder(z)
#         #     # get the mean of the negative binomial
#         #     px_rate = library * px_scale
#         #     # get the dispersion parameter
#         #     theta = torch.exp(self.log_theta)
#         #     # build count distribution
#         #     nb_logits = (px_rate + 1e-4).log() - (theta + 1e-4).log()
#         #     x_dist = dist.NegativeBinomial(total_count=theta, logits=nb_logits)
#         #     # score against actual counts
#         #     pyro.sample("obs", x_dist.to_event(1), obs=x)
#         pass
#
#     def guide(self, x, log_library):
#         # # define the guide (i.e. variational distribution) q(z|x)
#         # pyro.module("scvi", self)
#         # with pyro.plate("data", x.shape[0]):
#         #     # use the encoder to get the parameters used to define q(z|x)
#         #     x_ = torch.log(1 + x)
#         #     qz_m = self.mean_encoder(x_)
#         #     qz_v = self.var_encoder(x_)
#         #     # sample the latent code z
#         #     pyro.sample("latent", dist.Normal(qz_m, torch.sqrt(z_scale)).to_event(1))
#         pass
