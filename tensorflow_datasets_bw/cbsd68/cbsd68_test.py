"""bsds500 dataset."""

import tensorflow_datasets as tfds
from . import cbsd68


class CBSD68(tfds.testing.DatasetBuilderTestCase):
    """Tests for bsds500 dataset."""
    DATASET_CLASS = cbsd68.Cbsd68
    SPLITS = {
        'test': 2,  # Number of fake test example
    }
    DL_EXTRACT_RESULT = "."


if __name__ == '__main__':
    tfds.testing.test_main()
