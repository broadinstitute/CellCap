import torch
from torch import Tensor
import torch.nn.functional as F

from typing import Tuple


class DotProductAttention(torch.nn.Module):
    """
    Compute the dot products of the query with all values and apply a
    softmax function to obtain the weights on the values
    """

    def __init__(self, temp=1.0):
        super(DotProductAttention, self).__init__()
        self.temp = temp

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size, _, input_size = query.size(0), query.size(2), value.size(1)

        score = torch.bmm(query, key.transpose(1, 2))
        attn = F.softmax(score.view(-1, input_size) / self.temp, dim=1).view(
            batch_size, -1, input_size
        )
        context = torch.bmm(attn, value)

        return context, attn
