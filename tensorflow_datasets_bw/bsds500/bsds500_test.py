"""bsds500 dataset."""

import tensorflow_datasets as tfds
from . import bsds500


class Bsds500Test(tfds.testing.DatasetBuilderTestCase):
    """Tests for bsds500 dataset."""
    DATASET_CLASS = bsds500.Bsds500
    BUILDER_CONFIG_NAMES_TO_TEST = ['all']
    SPLITS = {
        'train': 2,  # Number of fake train example
        'test': 2,  # Number of fake test example
        'validation': 2,  # Number of fake val example
    }
    DL_EXTRACT_RESULT = "."


if __name__ == '__main__':
    tfds.testing.test_main()
