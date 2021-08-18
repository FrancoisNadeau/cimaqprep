#!usr/bin/env/python3

import setuptools
from setuptools import setup

setup(
    name="cimaqprep",
    version=0,
    author="francois",
    author_email="francois.nadeau.1@umontreal.ca",
    description="Functions to fetch a CIMA-Q participant's data",
    long_description="Functions to fetch a CIMA-Q participant data: Computing EPI Mask, clean fmri signal with according confounds, resample fMRI volumes to the number of Trials per task",
    long_description_content_type='text/x-rst',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    url="https://github.com/FrancoisNadeau/sniffbytes.git",
    python_requires=">=3.6",
)
