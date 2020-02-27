"""Test for the BSDS500 dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_datasets import testing
from tensorflow_datasets_bw import schelten_kernels


class ScheltenKernelsTest(testing.DatasetBuilderTestCase):
    EXAMPLE_DIR = 'tests/test_data/fake_examples/schelten_kernels'
    DL_DOWNLOAD_RESULT = 'kernels.mat'
    DATASET_CLASS = schelten_kernels.ScheltenKernels
    SPLITS = {
        "test": 192,  # Number of fake test example
    }
