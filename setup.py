#!/usr/bin/env python

import setuptools
from distutils.core import setup

dependencies = [
    "numpy",
    "scipy",
    "phylokrr",
    "joblib"
]

with open('README.md') as readme_file:
    readme = readme_file.read()

setup(name = "phyloK",
      version = '1.2.0',
    #   long_description = readme,
    #   long_description_content_type = 'text/markdown',
      packages = ['phyloK'],
      package_dir = {'phyloK': 'src'},
      python_requires='>=3.5',
      install_requires = dependencies,
      zip_safe = False,
      include_package_data=True,
      classifiers = [
          'Programming Language :: Python :: 3',
      ]
    )
