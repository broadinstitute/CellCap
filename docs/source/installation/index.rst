.. _installation:

Installation
============

Via pip
-------

Python packages can be conveniently installed from the Python Package Index (PyPI)
using `pip install <https://pip.pypa.io/en/stable/cli/pip_install/>`_.
CellCap is `available on PyPI <https://pypi.org/project/cellbender/>`_
and can be installed via

.. code-block:: console

  $ pip install cellcap

If your machine has a GPU with appropriate drivers installed, it should be
automatically detected, and the appropriate version of PyTorch with CUDA support
should automatically be downloaded as a CellBender dependency.

We recommend installing CellCap in its own
`conda environment <https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html>`_.
This allows for easier installation and prevents conflicts with any other python
packages you may have installed.

.. code-block:: console

  $ conda create -n cellcap python=3.9
  $ conda activate cellcap
  (cellcap) $ pip install cellcap


Installation from source
------------------------

Create a conda environment and activate it:

.. code-block:: console

  $ conda create -n cellcap python=3.9
  $ conda activate cellcap

Install the `pytables <https://www.pytables.org>`_ module:

.. code-block:: console

  (cellcap) $ conda install -c anaconda pytables

Install `pytorch <https://pytorch.org>`_ via
`these instructions <https://pytorch.org/get-started/locally/>`_:

.. code-block:: console

   (cellcap) $ pip install torch

and ensure that your installation is appropriate for your hardware (i.e. that
the relevant CUDA drivers get installed and that ``torch.cuda.is_available()``
returns ``True`` if you have a GPU available.

Clone this repository and install CellCap (in editable ``-e`` mode):

.. code-block:: console

   (cellcap) $ git clone https://github.com/broadinstitute/CellCap.git
   (cellcap) $ pip install -e CellCap