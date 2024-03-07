# This is rewriting of two classes in scvi-tools for purpose of using WeightedRandomSampler
# (AnnDataLoader and DataSplitter)

import copy
import numpy as np
from math import ceil, floor
from typing import Dict, List, Optional, Union

import lightning.pytorch as pl

import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    Sampler,
    WeightedRandomSampler,
)

from scvi import settings
from scvi.data import AnnDataManager


class AnnDataLoader(DataLoader):
    """DataLoader for loading tensors from AnnData objects.

    Parameters
    ----------
    adata_manager
        :class:`~scvi.data.AnnDataManager` object with a registered AnnData object.
    indices
        The indices of the observations in `adata_manager.adata` to load.
    batch_size
        Minibatch size to load each iteration. If `distributed_sampler` is `True`,
        refers to the minibatch size per replica. Thus, the effective minibatch
        size is `batch_size` * `num_replicas`.
    shuffle
        Whether the dataset should be shuffled prior to sampling.
    sampler
        Defines the strategy to draw samples from the dataset. Can be any Iterable with __len__ implemented.
        If specified, shuffle must not be specified. By default, we use a custom sampler that is designed to
        get a minibatch of data with one call to __getitem__.
    drop_last
        If `True` and the dataset is not evenly divisible by `batch_size`, the last
        incomplete batch is dropped. If `False` and the dataset is not evenly divisible
        by `batch_size`, then the last batch will be smaller than `batch_size`.
    drop_dataset_tail
        Only used if `distributed_sampler` is `True`. If `True` the sampler will drop
        the tail of the dataset to make it evenly divisible by the number of replicas.
        If `False`, then the sampler will add extra indices to make the dataset evenly
        divisible by the number of replicas.
    data_and_attributes
        Dictionary with keys representing keys in data registry (``adata_manager.data_registry``)
        and value equal to desired numpy loading type (later made into torch tensor) or list of
        such keys. A list can be used to subset to certain keys in the event that more tensors than
        needed have been registered. If ``None``, defaults to all registered data.
    iter_ndarray
        Whether to iterate over numpy arrays instead of torch tensors
    distributed_sampler
        ``EXPERIMENTAL`` Whether to use :class:`~scvi.dataloaders.BatchDistributedSampler` as the
        sampler. If `True`, `sampler` must be `None`.
    load_sparse_tensor
        ``EXPERIMENTAL`` If ``True``, loads data with sparse CSR or CSC layout as a
        :class:`~torch.Tensor` with the same layout. Can lead to speedups in data transfers to GPUs,
        depending on the sparsity of the data.
    **kwargs
        Additional keyword arguments passed into :class:`~torch.utils.data.DataLoader`.

    Notes
    -----
    If `sampler` is not specified, a :class:`~torch.utils.data.BatchSampler` instance is
    passed in as the sampler, which retrieves a minibatch of data with one call to
    :meth:`~scvi.data.AnnTorchDataset.__getitem__`. This is useful for fast access to
    sparse matrices as retrieving single observations and then collating is inefficient.
    """

    def __init__(
        self,
        adata_manager: AnnDataManager,
        indices: Optional[Union[list[int], list[bool]]] = None,
        batch_size: int = 128,
        shuffle: bool = False,
        sampler: Optional[Sampler] = None,
        drop_last: bool = False,
        # drop_dataset_tail: bool = False,
        data_and_attributes: Optional[Union[list[str], dict[str, np.dtype]]] = None,
        iter_ndarray: bool = False,
        weighted_sampler: bool = True,
        # load_sparse_tensor: bool = False,
        **kwargs,
    ):
        if indices is None:
            indices = np.arange(adata_manager.adata.shape[0])
        else:
            if hasattr(indices, "dtype") and indices.dtype is np.dtype("bool"):
                indices = np.where(indices)[0].ravel()
            indices = np.asarray(indices)
        self.indices = indices

        if weighted_sampler:
            self.sample_weights = torch.tensor(adata_manager.adata.obsm['X_weight'].squeeze())[indices]

        self.dataset = adata_manager.create_torch_dataset(
            indices=indices,
            data_and_attributes=data_and_attributes,
            # load_sparse_tensor=load_sparse_tensor,
        )
        if "num_workers" not in kwargs:
            kwargs["num_workers"] = settings.dl_num_workers

        self.kwargs = copy.deepcopy(kwargs)

        if sampler is not None and weighted_sampler:
            raise ValueError("Cannot specify both `sampler` and `weighted_sampler`.")

        # custom sampler for efficient minibatching on sparse matrices
        if sampler is None:
            if weighted_sampler:
                n_samples = len(indices)
                sampler = BatchSampler(
                    sampler=WeightedRandomSampler(self.sample_weights, n_samples),
                    batch_size=batch_size,
                    drop_last=drop_last,
                )
            else:
                sampler_cls = SequentialSampler if not shuffle else RandomSampler
                sampler = BatchSampler(
                    sampler=sampler_cls(self.dataset),
                    batch_size=batch_size,
                    drop_last=drop_last,
                )
            # do not touch batch size here, sampler gives batched indices
            # This disables PyTorch automatic batching, which is necessary
            # for fast access to sparse matrices
            self.kwargs.update({"batch_size": None, "shuffle": False})

        self.kwargs.update({"sampler": sampler})

        if iter_ndarray:
            self.kwargs.update({"collate_fn": lambda x: x})

        super().__init__(self.dataset, **self.kwargs)

def validate_data_split(
    n_samples: int, train_size: float, validation_size: Optional[float] = None
):
    """Check data splitting parameters and return n_train and n_val.

    Parameters
    ----------
    n_samples
        Number of samples to split
    train_size
        Size of train set. Need to be: 0 < train_size <= 1.
    validation_size
        Size of validation set. Need to be 0 <= validation_size < 1
    """
    if train_size > 1.0 or train_size <= 0.0:
        raise ValueError("Invalid train_size. Must be: 0 < train_size <= 1")

    n_train = ceil(train_size * n_samples)

    if validation_size is None:
        n_val = n_samples - n_train
    elif validation_size >= 1.0 or validation_size < 0.0:
        raise ValueError("Invalid validation_size. Must be 0 <= validation_size < 1")
    elif (train_size + validation_size) > 1:
        raise ValueError("train_size + validation_size must be between 0 and 1")
    else:
        n_val = floor(n_samples * validation_size)

    if n_train == 0:
        raise ValueError(
            "With n_samples={}, train_size={} and validation_size={}, the "
            "resulting train set will be empty. Adjust any of the "
            "aforementioned parameters.".format(n_samples, train_size, validation_size)
        )

    return n_train, n_val


class DataSplitter(pl.LightningDataModule):
    """Creates data loaders ``train_set``, ``validation_set``, ``test_set``.

    If ``train_size + validation_set < 1`` then ``test_set`` is non-empty.

    Parameters
    ----------
    adata_manager
        :class:`~scvi.data.AnnDataManager` object that has been created via ``setup_anndata``.
    train_size
        float, or None (default is 0.9)
    validation_size
        float, or None (default is None)
    shuffle_set_split
        Whether to shuffle indices before splitting. If `False`, the val, train, and test set are split in the
        sequential order of the data according to `validation_size` and `train_size` percentages.
    pin_memory
        Whether to copy tensors into device-pinned memory before returning them. Passed
        into :class:`~scvi.data.AnnDataLoader`.
    **kwargs
        Keyword args for data loader. If adata has labeled data, data loader
        class is :class:`~scvi.dataloaders.SemiSupervisedDataLoader`,
        else data loader class is :class:`~scvi.dataloaders.AnnDataLoader`.

    Examples
    --------
    >>> adata = scvi.data.synthetic_iid()
    >>> scvi.model.SCVI.setup_anndata(adata)
    >>> adata_manager = scvi.model.SCVI(adata).adata_manager
    >>> splitter = DataSplitter(adata)
    >>> splitter.setup()
    >>> train_dl = splitter.train_dataloader()
    """

    data_loader_cls = AnnDataLoader

    def __init__(
        self,
        adata_manager: AnnDataManager,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        shuffle_set_split: bool = True,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.adata_manager = adata_manager
        self.train_size = float(train_size)
        self.validation_size = validation_size
        self.shuffle_set_split = shuffle_set_split
        self.data_loader_kwargs = kwargs
        self.pin_memory = pin_memory or settings.dl_pin_memory_gpu_training

        self.n_train, self.n_val = validate_data_split(
            self.adata_manager.adata.n_obs, self.train_size, self.validation_size
        )

    def setup(self, stage: Optional[str] = None):
        """Split indices in train/test/val sets."""
        n_train = self.n_train
        n_val = self.n_val
        indices = np.arange(self.adata_manager.adata.n_obs)

        if self.shuffle_set_split:
            random_state = np.random.RandomState(seed=settings.seed)
            indices = random_state.permutation(indices)

        self.val_idx = indices[:n_val]
        self.train_idx = indices[n_val : (n_val + n_train)]
        self.test_idx = indices[(n_val + n_train) :]

    def train_dataloader(self):
        """Create train data loader."""
        return self.data_loader_cls(
            self.adata_manager,
            indices=self.train_idx,
            shuffle=True,
            drop_last=False,
            pin_memory=self.pin_memory,
            **self.data_loader_kwargs,
        )

    def val_dataloader(self):
        """Create validation data loader."""
        if len(self.val_idx) > 0:
            return self.data_loader_cls(
                self.adata_manager,
                indices=self.val_idx,
                shuffle=False,
                drop_last=False,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass

    def test_dataloader(self):
        """Create test data loader."""
        if len(self.test_idx) > 0:
            return self.data_loader_cls(
                self.adata_manager,
                indices=self.test_idx,
                shuffle=False,
                drop_last=False,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass