#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

setup(
    name="al_mlp",
    version="0.1",
    description="Active Learning for Machine Learning Potentials",
    author="Rui Qi Chen",
    author_email="ruiqic@andrew.cmu.edu",
    url="https://github.com/ulissigroup/al_mlp",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch",
        "spglib",
        "scikit-learn==0.21.3",
        "skorch==0.6.0",
        "ase",
        "scipy",
        "pandas",
    ],
)
