"""set5 dataset."""

import tensorflow_datasets as tfds
from . import set5


class Set5Test(tfds.testing.DatasetBuilderTestCase):
    """Tests for set5 dataset."""
    DATASET_CLASS = set5.Set5
    SPLITS = {
        'test': 2,  # Number of fake test example

    }
    DL_EXTRACT_RESULT = '.'


if __name__ == '__main__':
    tfds.testing.test_main()
