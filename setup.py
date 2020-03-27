#!/usr/bin/env python

from distutils.core import setup

setup(
    name='TensorFlow Datasets BW',
    version='0.1.1',
    description='Extensions to TensorFlow Datasets',
    author='Benjamin Wilhelm',
    author_email='benjamin.wilhelm@uni-konstanz.de',
    url='https://github.com/HedgehogCode/tensorflow-datasets-bw',
    packages=['tensorflow_datasets_bw'],
    install_requires=[
        'scipy',
        'tensorflow-datasets>=2',
    ],
    include_package_data=True,
)
