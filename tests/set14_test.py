"""Test for the Set5 dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_datasets import testing
from tensorflow_datasets_bw import set14


class Set14Test(testing.DatasetBuilderTestCase):
    EXAMPLE_DIR = 'tests/test_data/fake_examples/set14'
    DATASET_CLASS = set14.Set14
    SPLITS = {
        "test": 2,  # Number of fake test example
    }
