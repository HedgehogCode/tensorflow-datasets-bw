"""Test for the BSDS500 dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_datasets import testing
from tensorflow_datasets_bw import bsds500


class Bsds500Test(testing.DatasetBuilderTestCase):
    EXAMPLE_DIR = 'tests/test_data/fake_examples/bsds500'
    DATASET_CLASS = bsds500.Bsds500
    SPLITS = {
        "train": 2,  # Number of fake train example
        "test": 2,  # Number of fake test example
        "validation": 2,  # Number of fake val example
    }
