"""Helper for logging metrics."""

import torch


class Metrics:
    def __init__(self):
        self.metrics = {}

    def add(self, k, v):
        self.metrics.update({k: v})

    def remove(self, k):
        self.metrics.pop(k)

    def __iter__(self):
        for k, v in self.metrics.items():
            yield k, v


_METRICS_TO_LOG = Metrics()


def log_metric(name: str, tensor: torch.Tensor) -> None:
    """Log a value, saving it to the model history.
    Make use of logging in pytorch-lightning.

    The idea is to log scalars, not full tensors. This function accepts tensors
    and flattens them, assigning a unique name to each dimension. However, it
    is a bad idea to try to log a tensor with a very large number of elements.

    Parameters
    ----------
    name: Name of metric
    tensor: Value to log
    """
    tensor = tensor.detach()
    flat_tensor = tensor.flatten()
    if len(flat_tensor) > 10:
        raise UserWarning(
            f"You are logging a tensor with {len(tensor)} "
            f"elements. Each will have a separate column in "
            f"the trainer object history, with an underscore "
            f"appended to denote the entry."
        )

    if tensor.numel() == 1:
        _METRICS_TO_LOG.add(name, tensor)
    elif tensor.numel() > 1:
        for i, val in enumerate(flat_tensor):
            _METRICS_TO_LOG.add(f"{name}_{i}", val)
