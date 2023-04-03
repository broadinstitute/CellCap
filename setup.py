#!/usr/bin/env python

import os
import setuptools
from typing import List


def readme() -> str:
    with open("README.md") as f:
        return f.read()


def get_requirements() -> List[str]:
    filebase = os.path.dirname(__file__)

    def _readlines(filename):
        with open(os.path.join(filebase, filename)) as f:
            lines = f.readlines()
        return lines

    requirements = _readlines("requirements.txt")
    if "READTHEDOCS" in os.environ:
        requirements.extend(_readlines("requirements-rtd.txt"))
    if "DEV" in os.environ:
        requirements.extend(_readlines("requirements-dev.txt"))
    return requirements


def get_version() -> str:
    """Version number is centrally located in the file called VERSION"""
    with open(os.path.join(os.path.dirname(__file__), "VERSION")) as f:
        version = f.read().strip()
    return version


setuptools.setup(
    name="cellcap",
    version=get_version(),
    description="A software package for distilling interpretable insights from "
    "high-throughput single-cell RNA sequencing (scRNA-seq) "
    "perturbation experiments",
    long_description=readme(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research"
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords="scRNA-seq bioinformatics",
    url="https://github.com/broadinstitute/single-cell-compositional-perturbations",
    author="Yang Xu, Stephen Fleming, Mehrtash Babadi",
    license="BSD (3-Clause)",
    packages=setuptools.find_packages(),
    install_requires=get_requirements(),
    extras_require={
        "dev": ["pytest", "black==23.1.0", "flake8"],
    },
    # entry_points={
    #     'console_scripts': ['cellbender=cellbender.base_cli:main'],
    # },
    include_package_data=True,
    zip_safe=False,
)
