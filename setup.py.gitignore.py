# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:21:19 2024
@author: Klest Dedja
"""

raise RuntimeError(
    "This project uses pyproject.toml for configuration. Use `python -m build` instead of setup.py."
)

from pathlib import Path
from setuptools import find_packages, setup

readme_path = Path("README.md")
with readme_path.open(encoding="utf-8") as f:
    long_description = f.read()

version_file_path = Path("app") / "bellatrex" / "version.txt"
with version_file_path.open(encoding="utf-8") as version_file:
    version = version_file.read().strip()

setup(
    name="bellatrex",
    version=version,
    description="A toolbox for Building Explanations through a LocaLly AccuraTe Rule EXtractor",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Klest94/Bellatrex",
    author="Klest Dedja",
    author_email="daneel.olivaw94@gmail.com",
    license="MIT",
    # Find package source under 'app/'
    package_dir={"": "app"},
    packages=find_packages(where="app"),
    include_package_data=True,  # Includes files from MANIFEST.in
    package_data={
        "bellatrex.datasets": ["*.csv"],  # Include bundled datasets
    },
    install_requires=[
        "scikit-learn >= 1.2",
        "threadpoolctl>=3.1",
        "scikit-survival>=0.22, <1.0",
        "scipy>=1.11",
        "pandas>=1.5",
        "matplotlib>=3.7",
    ],
    extras_require={
        "dev": ["pytest", "twine"],
        "gui": ["dearpygui>=1.6.2, <2.0", "dearpygui-ext>=0.9.5, <1.0"],
    },
    python_requires=">=3.9, <3.13",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        # Add these once tested:
        # "Operating System :: Microsoft :: Windows",
        # "Operating System :: POSIX :: Linux",
        # "Operating System :: MacOS :: MacOS X",
    ],
)
