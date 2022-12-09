"""Basic components for neural networks."""

import torch

from typing import List, Optional


class FullyConnectedLayer(torch.nn.Module):
    """Neural network unit made of a fully connected linear layer, but
    customizable including shapes, activations, batch norm, layer norm, and
    dropout.

    Parameters
    ----------
    input_dim
        Number of features for input
    output_dim
        Number of features for output
    activation
        Activation function to be applied to each hidden layer
        (default :py:class:`torch.nn.ReLU`)
    use_batch_norm
        True to apply batch normalization using :py:class:`torch.nn.BatchNorm1d`
        with ``momentum=0.01``, ``eps=0.001``
        (default False)
    use_layer_norm
        True to apply layer normalization (after optional batch normalization)
        using :py:class:`torch.nn.LayerNorm` with ``elementwise_affine=False``
        (default False)
    dropout_rate
        Dropout rate to use in :py:class:`torch.nn.Dropout` before linear layer

    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 activation: torch.nn.Module,
                 use_batch_norm: bool = False,
                 use_layer_norm: bool = False,
                 dropout_rate: Optional[float] = None):

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # set up layers as a list of Linear modules with appropriate extras
        modules = []
        if dropout_rate is not None:
            modules.append(torch.nn.Dropout(p=dropout_rate))
        modules.append(torch.nn.Linear(in_features=input_dim, out_features=output_dim))
        if use_batch_norm:
            modules.append(torch.nn.BatchNorm1d(num_features=output_dim, momentum=0.01, eps=0.001))
        if use_layer_norm:
            modules.append(torch.nn.LayerNorm(normalized_shape=output_dim, elementwise_affine=False))
        if activation is not None:
            modules.append(activation)

        # concatenate Linear layers using Sequential
        self.layer = torch.nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class FullyConnectedNetwork(torch.nn.Module):
    """Neural network made of fully connected linear layers,
    :py:class:`FullyConnectedLayer`. Architecture is customizable including
    shapes, activations, batch norm, layer norm, and dropout.

    Parameters
    ----------
    input_dim
        Number of features for input
    hidden_dims
        List of hidden layer sizes, can be empty list []
    output_dim
        Number of features for output
    hidden_activation
        Activation function to be applied to each hidden layer
        (default :py:class:`torch.nn.ReLU`)
    output_activation
        Activation function to be applied to output (default None)
    use_batch_norm
        True to apply batch normalization using :py:class:`torch.nn.BatchNorm1d`
        with ``momentum=0.01``, ``eps=0.001``
        (default False)
    use_layer_norm
        True to apply layer normalization (after optional batch normalization)
        using :py:class:`torch.nn.LayerNorm` with ``elementwise_affine=False``
        (default False)
    dropout_rate
        Dropout rate to use in :py:class:`torch.nn.Dropout` for each hidden
        layer (applied before each layer)

    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 hidden_activation: torch.nn.Module = torch.nn.ReLU(),
                 output_activation: Optional[torch.nn.Module] = None,
                 use_batch_norm: bool = False,
                 use_layer_norm: bool = False,
                 dropout_rate: Optional[float] = None):

        super().__init__()

        # set up layers as a list of Linear modules with appropriate extras
        layers = [FullyConnectedLayer(input_dim=i,
                                      output_dim=j,
                                      activation=hidden_activation,
                                      use_batch_norm=use_batch_norm,
                                      use_layer_norm=use_layer_norm,
                                      dropout_rate=dropout_rate)
                  for i, j in zip([input_dim] + hidden_dims, hidden_dims)]
        dim = input_dim if (len(hidden_dims) == 0) else layers[-1].output_dim
        layers.append(FullyConnectedLayer(input_dim=dim,
                                          output_dim=output_dim,
                                          activation=output_activation,
                                          use_batch_norm=False,
                                          use_layer_norm=False,
                                          dropout_rate=dropout_rate))

        # concatenate Linear layers using Sequential
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ReverseGradientFunction(torch.autograd.Function):
    """Gradient reversal :py:class:`torch.autograd.Function`

    Credit to https://github.com/janfreyberg/pytorch-revgrad
    """

    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


class GradientReversal(torch.nn.Module):
    """A gradient reversal layer as a :py:class:`torch.nn.Module`.
    This module has no parameters, and simply reverses the gradient
    in the backward pass.

    Parameters
    ----------
    alpha
        Positive multiplicative factor to be applied at the time of gradient reversal

    Credit to https://github.com/janfreyberg/pytorch-revgrad
    """

    def __init__(self, alpha=1.):
        assert alpha > 0, 'Multiplier `alpha` must be positive'
        super().__init__()
        self._alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, input_):
        return ReverseGradientFunction.apply(input_, self._alpha)


class TorchExponential(torch.nn.Module):
    """:py:class:`torch.exp` as a :py:class:`torch.nn.Module` so that it can
    be used as an (output) activation function in a network
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)
