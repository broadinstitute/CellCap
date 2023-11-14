CellCap
==========

CellCap is a variational autoencoder for modeling correspondence between cellular identity and perturbation response
in single-cell data.

Installation
----------------------

CellCap can be installed via

.. code-block:: console

  $ pip install cellcap

(and we recommend installing in its own ``conda`` environment to prevent
conflicts with other software).


Advanced installation
---------------------

From source for development
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a conda environment and activate it:

.. code-block:: console

  $ conda create -n cellcap python=3.9
  $ conda activate cellcap

Install `pytorch <https://pytorch.org>`_ via
`these instructions <https://pytorch.org/get-started/locally/>`_, for example:

.. code-block:: console

   (cellcap) $ pip install torch

and ensure that your installation is appropriate for your hardware (i.e. that
the relevant CUDA drivers get installed and that ``torch.cuda.is_available()``
returns ``True`` if you have a GPU available.

Clone this repository and install CellBender (in editable ``-e`` mode):

.. code-block:: console

   (cellcap) $ git clone https://github.com/broadinstitute/CellCap.git
   (cellcap) $ pip install -e CellCap

Citing CellCap
-----------------

If you use CellCap in your research, please consider citing our paper:

Yang Xu, Stephen Fleming, Matthew Tegtmeyer, Steven A. McCarroll, and Mehrtash Babadi.
Modeling interpretable correspondence between cellular identity and perturbation response with CellCap.
*bioRxiv*, 2023. https://doi.org/10.1038/s41592-023-01943-7
