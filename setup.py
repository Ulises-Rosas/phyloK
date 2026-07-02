#!/usr/bin/env python

import setuptools
from distutils.core import setup

dependencies = [
    "numpy"
]

with open('README.md') as readme_file:
    readme = readme_file.read()

setup(name = "phylokrr_dev",
      version = '0.1.0',
    #   long_description = readme,
    #   long_description_content_type = 'text/markdown',
      packages = ['phylokrr_dev'],
      package_dir = {'phylokrr_dev': 'src'},
      python_requires='>=3.5',
      install_requires = dependencies,
      zip_safe = False,
      include_package_data=True,
    #   package_data={'phylokrr': ['data/*.csv']},
      classifiers = [
          'Programming Language :: Python :: 3',
      ]
    )
