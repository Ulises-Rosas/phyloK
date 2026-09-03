#!/usr/bin/env python

import setuptools
from distutils.core import setup

dependencies = [
    "numpy",
    "scipy"
]

with open('README.md') as readme_file:
    readme = readme_file.read()

setup(name = "phylokrr_dev",
      version = '0.8.1',
    #   long_description = readme,
    #   long_description_content_type = 'text/markdown',
      packages = ['phylokrr_dev'],
      package_dir = {'phylokrr_dev': 'src'},
      python_requires='>=3.5',
      install_requires = dependencies,
      zip_safe = False,
      include_package_data=True,
      classifiers = [
          'Programming Language :: Python :: 3',
      ]
    )
