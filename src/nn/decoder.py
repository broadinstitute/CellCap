"""Decoder neural network(s)"""

import torch

from .base import FullyConnectedLayer

from typing import Optional, Dict


class LinearDecoder(torch.nn.Module):
    """Linear decoder with no hidden layers.  Creates a single output value.

    Parameters
    ----------
    input_dim
        Number of features for input
    output_dim
        Number of features for output
    output_activation
        Activation function to be applied to output (default None)

    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 output_activation: Optional[torch.nn.Module] = None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.network = FullyConnectedLayer(
            input_dim=input_dim,
            output_dim=output_dim,
            activation=output_activation,
            use_batch_norm=False,
            use_layer_norm=False,
            dropout_rate=None,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Send data through decoder.

        Args:
            x: Data

        Returns:
            Output :py:class:`torch.Tensor`

        """
        return self.network(x)
