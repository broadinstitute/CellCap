"""Encoder neural network(s)"""

import torch

from .base import FullyConnectedNetwork, FullyConnectedLayer, TorchExponential

from typing import List, Optional, Dict


class Encoder(torch.nn.Module):
    """Encoder for a VAE where we need a mean and a variance for a distribution
    in latent space.

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
    loc_output_activation
        Activation function to be applied to "loc" output (default None)
    scale_output_activation
        Activation function to be applied to "scale" output
        (default :py:class:`~src.nn.base.TorchExponential`)
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
                 loc_output_activation: Optional[torch.nn.Module] = None,
                 scale_output_activation: Optional[torch.nn.Module] = TorchExponential(),
                 use_batch_norm: bool = False,
                 use_layer_norm: bool = False,
                 dropout_rate: Optional[float] = None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if len(hidden_dims) == 0:
            self.network = lambda x: x
            readout_dim = input_dim
        else:
            self.network = FullyConnectedNetwork(
                input_dim=input_dim,
                hidden_dims=hidden_dims[:-1],
                output_dim=hidden_dims[-1],
                hidden_activation=hidden_activation,
                output_activation=hidden_activation,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                dropout_rate=dropout_rate,
            )
            readout_dim = hidden_dims[-1]

        self.loc_readout = FullyConnectedLayer(
            input_dim=readout_dim,
            output_dim=output_dim,
            activation=loc_output_activation,
            use_batch_norm=False,
            use_layer_norm=False,
            dropout_rate=dropout_rate,
        )

        self.scale_readout = FullyConnectedLayer(
            input_dim=readout_dim,
            output_dim=output_dim,
            activation=scale_output_activation,
            use_batch_norm=False,
            use_layer_norm=False,
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Send data through encoder.

        Args:
            x: Data

        Returns:
            Dict with keys ["loc", "scale"], each value a :py:class:`torch.Tensor`

        """
        hidden = self.network(x)
        return {'loc': self.loc_readout(hidden),
                'scale': self.scale_readout(hidden)}
