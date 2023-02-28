"""Gradient reversal functionality
from https://github.com/janfreyberg/pytorch-revgrad/blob/master/src/pytorch_revgrad/module.py
"""

import torch


class GradientReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


class GradientReverseModule(torch.nn.Module):
    def __init__(self, alpha=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, input_):
        return GradientReverseLayer.apply(input_, self._alpha)
