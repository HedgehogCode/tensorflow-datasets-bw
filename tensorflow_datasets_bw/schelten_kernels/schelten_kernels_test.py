"""schelten_kernels dataset."""

import tensorflow_datasets as tfds
from . import schelten_kernels


class ScheltenKernelsTest(tfds.testing.DatasetBuilderTestCase):
    """Tests for schelten_kernels dataset."""
    # TODO do not use the full dataset for testing
    DATASET_CLASS = schelten_kernels.ScheltenKernels
    BUILDER_CONFIG_NAMES_TO_TEST = ['all']
    SPLITS = {
        'test': 192,  # Number of fake test example
    }
    DL_EXTRACT_RESULT = 'kernels.mat'


if __name__ == '__main__':
    tfds.testing.test_main()
