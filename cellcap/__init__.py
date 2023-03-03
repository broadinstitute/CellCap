"""Namespace for the package"""

from .scvi_module import CellCap
from .utils import generate_simulated_dataset


__all__ = [
    CellCap,
    generate_simulated_dataset,
]
