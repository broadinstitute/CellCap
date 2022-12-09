"""Neural network(s) for the adversarial classification task"""

import torch

from .base import FullyConnectedNetwork

from typing import List, Optional


class Classifier(torch.nn.Module):
    """Classifier network that outputs a value for each class.

    Default is to use no output activation function, and so the outputs could be
    used as un-normalized logit probabilities (appropriate for
    :py:class:`torch.nn.CrossEntropyLoss`).

    Does not include gradient reversal.

    Parameters
    ----------
    input_dim
        Number of features for input
    hidden_dims
        List of hidden layer sizes, can be empty list []
    output_dim
        Number of features for output, i.e. number of classes
    hidden_activation
        Activation function to be applied to each hidden layer
        (default :py:class:`torch.nn.ReLU`)
    output_activation
        Activation function to be applied to "loc" output (default None)

    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 hidden_activation: torch.nn.Module = torch.nn.ReLU(),
                 output_activation: Optional[torch.nn.Module] = None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.network = FullyConnectedNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            use_batch_norm=False,
            use_layer_norm=False,
            dropout_rate=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Send data through classifier network.

        Args:
            x: Data

        Returns:
            :py:class:`torch.Tensor`

        """
        return self.network(x)
